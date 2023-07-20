import gym
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys, os
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import gridworld_vision
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        # self.fc1 = nn.Linear(4, 128)
        # self.fc2 = nn.Linear(128, 2)
        self.fc1 = nn.Linear(8, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 4)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    #env = gym.make('CartPole-v1', render_mode="human")
    env = gridworld_vision.GridworldEnv(5,10)
    pi = Policy()
    score = 0.0
    print_interval = 1
    epi_num = 2100
    for n_epi in range(epi_num):

        s = env.reset()
        s = np.delete(s, s.size-1)
        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            if n_epi > epi_num-3:
                env._render()
                time.sleep(0.1)
            #prob = pi(torch.from_numpy(s).float()) # prob : 4개 공간
            prob = pi(torch.from_numpy(s).float()) # prob : 4개 공간
            m = Categorical(prob) # 4개중 하나 sampling
            a = m.sample() # a : sampling 된 action

            s_prime, r, done, _ = env.step(a.item()) # a.item() : action의 숫자값 (0,1,2,3)
            pi.put_data((r,prob[a]))
            s = np.delete(s_prime, s_prime.size-1)
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            writer.add_scalar('Loss/episode', score/print_interval, n_epi)
            score = 0.0
    writer.close()
    torch.save(pi.state_dict(), ".\weights\model_state_dict.pt")
    torch.save(pi, ".\weights\model.pt")


if __name__ == '__main__':
    main()