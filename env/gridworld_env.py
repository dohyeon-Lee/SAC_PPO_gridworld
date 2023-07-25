from gym import Env
from gym.utils import seeding
import numpy as np
import io
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import gridworld_observation

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(Env):
        
    def __init__(self, flag = "fix"):
        self.obs = gridworld_observation.Observation()
        self.seed()
        self.reset(flag)
        
    def reset(self, flag):
        self.make_MAP(flag)
        self.state, self.vision_pos = self.obs.reset(self.inital_pos, self.terminal_pos, self.mine_pos, self.MAX_X, self.MAX_Y)
        self.nA = 4
        self.npos = self.MAX_X * self.MAX_Y
        self.shape = [self.MAX_Y, self.MAX_X]
        self.move_count = 0
        self.collision = 0
        self.pos = self.inital_pos # 현재 위치
        self.P = self.make_prob()
        return self.state
    
    def step(self, a): # imput : action output : next state, reward
        next_pos, agent_condition, d = self.P[self.pos][a]
        self.pos = next_pos
        self.state, self.vision_pos = self.obs.step(self.pos, a)
        ################################################################
        #### make reward ###############################################
        reward = 0
        if d == True : 
            #reward = 1000*(self.npos/self.move_count)
            collision_reward = 10000 - self.collision #10000/(0.01+self.collision)
            reward = collision_reward
            print("# of hit wall : {}, self collision reward : {} ".format(self.collision, collision_reward))
            self.collision = 0
            self.move_count = 0
        elif agent_condition[a] == -1 : #move
            dx = self.state[0]
            dy = self.state[1]
            reward = -np.abs(dx)-np.abs(dy)
            #print("distance : {}, reward : {}".format(distance, reward))
        elif agent_condition[a] == -2 : #hit wall
            dx = self.state[0]
            dy = self.state[1]
            reward = -np.abs(dx)-np.abs(dy)
            reward += -10
            self.collision += 1            
        self.move_count += 1

        return (self.state, reward*0.01, d)

    def _render(self, mode='human', close=False):

        os.system('cls')
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.npos).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            if self.pos == s:
                output = "🔴"
            elif s == self.terminal_pos:
                output = "🟪"
            elif s == self.inital_pos:
                output = "🟩"
            elif s in self.vision_pos:
                output = "🟡"
            elif s in self.mine_pos:
                output = "🟦"
            else:
                output = "🟨"
            #######

            if x == 0:
                output = output.lstrip()
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def cal_pos(self, index, MAX_X):
        return (index[1] * MAX_X) + index[0]

    def make_MAP(self, flag = "fix"):
        #####################################################
        #### define possible inital pos & final pos #########
        if flag == "random" :
            self.MAX_X = np.random.randint(low=10,high=40)
            self.MAX_Y = np.random.randint(low=10,high=40)
        else :
            self.MAX_X = 10
            self.MAX_Y = 10
        index_1 = [0,0]
        index_2 = [self.MAX_X-1, self.MAX_Y-1]
        index_3 = [0, self.MAX_Y-1]
        index_4 = [self.MAX_X-1, 0]
        index = [index_1, index_2, index_3, index_4]
        pos_1 = self.cal_pos(index_1,self.MAX_X)
        pos_2 = self.cal_pos(index_2,self.MAX_X)
        pos_3 = self.cal_pos(index_3,self.MAX_X)
        pos_4 = self.cal_pos(index_4,self.MAX_X)
        posset_1 = [pos_1, pos_2]
        posset_2 = [pos_3, pos_4]
        pos = [posset_1, posset_2]
        #####################################################
        #### define inital pos ##############################
        inital_pos_idx = np.random.randint(low=0, high=2)
        terminate_pos_idx = np.random.randint(low=0, high=2)
        
        setnum = np.random.randint(low=0, high=2)
        
        while True : 
            if inital_pos_idx == terminate_pos_idx:
                terminate_pos_idx = np.random.randint(low=0, high=2)
            else: 
                break
        self.inital_pos = pos[setnum][inital_pos_idx]
        self.terminal_pos = pos[setnum][terminate_pos_idx]
        #####################################################
        #### MAP MAKING #####################################
        def mine_grid(mine_num, index):
            x_list = np.random.randint(low=0, high=self.MAX_X, size=mine_num)
            y_list = np.random.randint(low=0, high=self.MAX_Y, size=mine_num)
            x_list = np.expand_dims(x_list, axis=0)
            y_list = np.expand_dims(y_list, axis=0)
            list = np.append(x_list, y_list, axis=0)
            list = list.T

            mine_list = list
            for i in range(mine_num):
                for j in range(4):
                    if(list[i,0] == index[j][0] and list[i,1] == index[j][1]):
                        mine_list = np.delete(list, i, axis = 0)

            return mine_list 
        
        if flag == "random":                
            mine_num = 50
            self.mine_index = mine_grid(mine_num, index)
            mine_num = self.mine_index.shape[0] # random으로 생성된 mine 중 시작점, 끝점과 겹치는 경우 제거했기에 mine 수 재조정
        else : 
            mine_num = 36
            self.mine_index = np.zeros((mine_num,2))
            # self.mine_index[0,:] = [1, 2]
            # self.mine_index[1,:] = [1, 4]
            # self.mine_index[2,:] = [3, 0]
            # self.mine_index[3,:] = [2, 2]
            # self.mine_index[4,:] = [4, 3]
      
            # self.mine_index[0,:] = [0, 1]
            # self.mine_index[1,:] = [1, 1]
            # self.mine_index[2,:] = [2, 1]
            # self.mine_index[3,:] = [2, 2]
            # self.mine_index[4,:] = [3, 3]
            # self.mine_index[5,:] = [4, 3]
            # self.mine_index[6,:] = [3, 4]
            # self.mine_index[7,:] = [6, 7]
            # self.mine_index[8,:] = [3, 7]
            # self.mine_index[9,:] = [9, 3]
            # self.mine_index[10,:] = [5, 3]
            # self.mine_index[11,:] = [1, 6]
            # self.mine_index[12,:] = [5, 6]
            # self.mine_index[13,:] = [7, 3]
            # self.mine_index[14,:] = [8, 7]
            # self.mine_index[15,:] = [2, 6]
            # self.mine_index[16,:] = [8, 5]
            # self.mine_index[17,:] = [8, 2]
            # self.mine_index[18,:] = [9, 1]
            # self.mine_index[19,:] = [6, 1]
            # self.mine_index[20,:] = [4, 1]
            # self.mine_index[21,:] = [4, 2]
            # self.mine_index[22,:] = [7, 6]
            # self.mine_index[23,:] = [4, 9]
            # self.mine_index[24,:] = [4, 8]
            # self.mine_index[25,:] = [6, 9]

            self.mine_index[0,:] = [1, 2] #
            self.mine_index[1,:] = [1, 2]
            self.mine_index[2,:] = [2, 1]
            self.mine_index[3,:] = [2, 1] #
            self.mine_index[4,:] = [4, 1]
            self.mine_index[5,:] = [4, 2]
            self.mine_index[6,:] = [5, 1]
            self.mine_index[7,:] = [5, 2]
            self.mine_index[8,:] = [7, 1]
            self.mine_index[9,:] = [8, 2] #
            self.mine_index[10,:] = [8, 2] #
            self.mine_index[11,:] = [8, 2]
            self.mine_index[12,:] = [1, 4]
            self.mine_index[13,:] = [1, 5]
            self.mine_index[14,:] = [2, 4]
            self.mine_index[15,:] = [2, 5]
            self.mine_index[16,:] = [4, 4]
            self.mine_index[17,:] = [4, 5]
            self.mine_index[18,:] = [5, 4]
            self.mine_index[19,:] = [5, 5]
            self.mine_index[20,:] = [7, 4]
            self.mine_index[21,:] = [7, 5]
            self.mine_index[22,:] = [8, 4]
            self.mine_index[23,:] = [8, 5]
            self.mine_index[24,:] = [1, 7]
            self.mine_index[25,:] = [1, 7] #
            self.mine_index[26,:] = [2, 8] #
            self.mine_index[27,:] = [2, 8]
            self.mine_index[28,:] = [4, 7]
            self.mine_index[29,:] = [4, 8]
            self.mine_index[30,:] = [5, 7]
            self.mine_index[31,:] = [5, 8]
            self.mine_index[32,:] = [7, 8] # 
            self.mine_index[33,:] = [7, 8]
            self.mine_index[34,:] = [8, 7]
            self.mine_index[35,:] = [8, 7] #


        self.mine_pos = np.zeros(mine_num)
        for i in range(mine_num):
            self.mine_pos[i] = self.cal_pos(self.mine_index[i],self.MAX_X)

        # mapdata : inital_pos, terminal_pos, mine_pos  

    def make_prob(self):
        P = {} # make next state
        grid = np.arange(self.npos).reshape(self.shape)
        it = np.nditer(grid, flags=["multi_index"])
        
        while not it.finished:
            pos = it.iterindex
            y, x = it.multi_index
            

            P[pos] = {a: [] for a in range(self.nA)} # 각 position state별로 가능한 action 수 (4개) 공간 만들기
            
            def is_done(pos):
                if pos == self.terminal_pos:
                    return True
                else:
                    return False

            agent_condition = -1 * np.ones(self.nA)

            if y == 0 : # 벽이나 낭떠러지(바깥쪽) 이동하려 할 시
                ns_up_1 = pos
                agent_condition[0] = -2
            elif (pos - self.MAX_X in self.mine_pos) :
                ns_up_1 = pos
                agent_condition[0] = -2
            else : 
                ns_up_1 = pos - self.MAX_X
            if x == (self.MAX_X - 1) :
                ns_right_1 = pos
                agent_condition[1] = -2
            elif (pos + 1 in self.mine_pos) :
                ns_right_1 = pos
                agent_condition[1] = -2 
            else : 
                ns_right_1 = pos + 1
            if y == (self.MAX_Y - 1) :
                ns_down_1 = pos
                agent_condition[2] = -2
            elif (pos + self.MAX_X in self.mine_pos) :
                ns_down_1 = pos
                agent_condition[2] = -2
            else :
                ns_down_1 = pos + self.MAX_X
            if x == 0 :
                ns_left_1 = pos
                agent_condition[3] = -2
            elif (pos - 1 in self.mine_pos) :
                ns_left_1 = pos
                agent_condition[3] = -2
            else : 
                ns_left_1 = pos - 1
              
            P[pos][UP] = [ns_up_1, agent_condition, is_done(ns_up_1)]
            P[pos][RIGHT] = [ns_right_1, agent_condition, is_done(ns_right_1)]
            P[pos][DOWN] = [ns_down_1, agent_condition, is_done(ns_down_1)]
            P[pos][LEFT] = [ns_left_1, agent_condition, is_done(ns_left_1)]

            it.iternext()
        return P

def main():
    env = GridworldEnv("fix")
    env._render()
    
if __name__ == "__main__":
    main()