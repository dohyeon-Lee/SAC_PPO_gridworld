import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random
from torch.distributions import Categorical


import sys, os
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import gridworld_env
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


#Hyperparameters
lr_pi           = 0.0005
lr_q            = 0.001
init_alpha      = 0.01
gamma           = 0.98
batch_size      = 32
buffer_limit    = 50000
tau             = 0.01 # for target network soft update
target_entropy  = -1.0 # for automated alpha update
lr_alpha        = 0.001  # for automated alpha update

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0 
            done_mask_lst.append([done_mask])
        
        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst, dtype=torch.float), \
                torch.tensor(r_lst, dtype=torch.float), torch.tensor(s_prime_lst, dtype=torch.float), \
                torch.tensor(done_mask_lst, dtype=torch.float)
    
    def size(self):
        return len(self.buffer)

class PolicyNet(nn.Module):
    def __init__(self, learning_rate):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(8, 256)
        self.fc_pi = nn.Linear(256,4)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.alpha = init_alpha
        # self.log_alpha = torch.tensor(np.log(init_alpha))
        # self.log_alpha.requires_grad = True
        # self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        m = Categorical(prob)
        action = m.sample()
        
        action = action.unsqueeze(-1)
        if action.dim() == 1:
            prob = prob[action]
        else:
            # 행 인덱스를 생성합니다.
            rows = torch.arange(0, prob.size(0)).unsqueeze(1)

            # 행 인덱스와 열 인덱스를 합쳐서 최종 인덱스를 만듭니다.
            indices = torch.cat((rows, action), dim=1)

            # 각 샘플에 해당하는 확률을 가져옵니다.
            prob = prob[indices[:, 0], indices[:, 1]].unsqueeze(1)
        
        #print(prob)
        #log_prob = torch.log(prob)
        #real_action = torch.tanh(action)
        #real_log_prob = log_prob # - torch.log(1-torch.tanh(action).pow(2) + 1e-7)
        return action, prob

    def train_net(self, q1, q2, mini_batch):
        
        s, _, _, _, _ = mini_batch
        a, prob = self.forward(s, softmax_dim=1)
        entropy = - self.alpha * torch.log(prob)
        # entropy = -self.log_alpha.exp() * log_prob

        q1_val, q2_val = q1(s,a), q2(s,a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]

        loss = -min_q - entropy # for gradient ascent , loss_pi , 8
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

        # self.log_alpha_optimizer.zero_grad()
        # alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()
        # alpha_loss.backward()
        # self.log_alpha_optimizer.step()

class QNet(nn.Module):
    def __init__(self, learning_rate):
        super(QNet, self).__init__()
        self.fc_s = nn.Linear(8, 64)
        self.fc_a = nn.Linear(1,64)
        self.fc_cat = nn.Linear(128,32)
        self.fc_out = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x, a):
        a = a.to(torch.float)
        h1 = F.relu(self.fc_s(x))
        h2 = F.relu(self.fc_a(a))
        cat = torch.cat([h1,h2], dim=1)
        q = F.relu(self.fc_cat(cat))
        q = self.fc_out(q)
        return q

    def train_net(self, target, mini_batch):
        
        s, a, r, s_prime, done = mini_batch
        
        loss = F.smooth_l1_loss(self.forward(s, a) , target) # 7
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

    def soft_update(self, net_target): # 9
        for param_target, param in zip(net_target.parameters(), self.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - tau) + param.data * tau)

def calc_target(pi, q1, q2, mini_batch): # 6
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        # print(s_prime.dim())
        a_prime, prob= pi.forward(s_prime, softmax_dim=1)
        # print("a: {}, p: {}".format(a_prime.dim(), prob.dim()))
        # print(a_prime)
        entropy = -pi.alpha * torch.log(prob)
        #entropy = -pi.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime,a_prime), q2(s_prime,a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + gamma * done * (min_q + entropy)

    return target
    
def main():
    flag = "fix" # "fix":training  "random":playing 
    env = gridworld_env.GridworldEnv(flag)
    memory = ReplayBuffer()
    q1, q2, q1_target, q2_target = QNet(lr_q), QNet(lr_q), QNet(lr_q), QNet(lr_q)
    pi = PolicyNet(lr_pi)

    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())
    arguments = sys.argv

    
    if len(arguments) > 1:
        if sys.argv[1] == "continue":
            pi.load_state_dict(torch.load(".\weights\model_state_dict.pt"))  
    print_interval = 1
    score = 0.0
    epi_num = 10000
    for n_epi in range(epi_num):
        s = env.reset(flag)
        done = False
        count = 0
        if n_epi > epi_num - 10:
            env._render()
            time.sleep(0.5)
        while count < 200 and not done:
            a, prob= pi.forward(torch.from_numpy(s).float())
            s_prime, r, done = env.step(a.item())
            memory.put((s, a.item(), r/10.0, s_prime, done))
            score +=r
            s = s_prime
            count += 1
        
        if memory.size()>1000:
            for i in range(20):
                mini_batch = memory.sample(batch_size)
                td_target = calc_target(pi, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                entropy = pi.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)
                
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f} ".format(n_epi, score/print_interval))
            writer.add_scalar('return/episode', 10*score/print_interval, n_epi)
            score = 0.0

    writer.close()
    torch.save(pi.state_dict(), ".\weights\model_state_dict_pi.pt")
    torch.save(pi, ".\weights\model_pi.pt")
if __name__ == '__main__':
    main()