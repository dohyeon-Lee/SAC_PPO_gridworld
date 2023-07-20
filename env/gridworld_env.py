from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text.utils import categorical_sample
import numpy as np
from gridworld_observation import Observation
class DiscreteEnv(Env):

    def make_MAP(self, flag = "fix"):
        #####################################################
        #### define possible inital pos & final pos #####
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
        pos = [pos_1, pos_2, pos_3, pos_4]
        #####################################################
        #### define inital pos ############################
        inital_pos_idx = np.random.randint(low=0, high=4)
        terminate_pos_idx = np.random.randint(low=0, high=4)
        while True : 
            if inital_pos_idx == terminate_pos_idx:
                terminate_pos_idx = np.random.randint(low=0, high=4)
            else: 
                break
        self.inital_pos = pos[inital_pos_idx]
        self.terminal_pos = pos[terminate_pos_idx]
        #####################################################
        #### MAP MAKING #####################################
        def mine_grid(self, mine_num, index):
            x_list = np.random.randint(low=0, high=self.MAX_X, size=mine_num)
            y_list = np.random.randint(low=0, high=self.MAX_Y, size=mine_num)
            x_list = np.expand_dims(x_list, axis=0)
            y_list = np.expand_dims(y_list, axis=0)
            list = np.append(x_list, y_list, axis=0)
            list = list.T

            mine_list = list
            for i in range(mine_num):
                if(list[i] in index):
                    mine_list = np.delete(list, i, axis = 0)       
            return mine_list 
        
        if flag == "random":                
            mine_num = 100
            self.mine_index = mine_grid(mine_num, index)
            mine_num = self.mine_index.shape[0] # random으로 생성된 mine 중 시작점, 끝점과 겹치는 경우 제거했기에 mine 수 재조정
        else : 
            mine_num = 26
            self.mine_index = np.zeros((mine_num,2))
            self.mine_index[0,:] = [0, 1]
            self.mine_index[1,:] = [1, 1]
            self.mine_index[2,:] = [2, 1]
            self.mine_index[3,:] = [2, 2]
            self.mine_index[4,:] = [3, 3]
            self.mine_index[5,:] = [4, 3]
            self.mine_index[6,:] = [3, 4]
            self.mine_index[7,:] = [6, 7]
            self.mine_index[8,:] = [3, 7]
            self.mine_index[9,:] = [9, 3]
            self.mine_index[10,:] = [5, 3]
            self.mine_index[11,:] = [1, 6]
            self.mine_index[12,:] = [5, 6]
            self.mine_index[13,:] = [7, 3]
            self.mine_index[14,:] = [8, 7]
            self.mine_index[15,:] = [2, 6]
            self.mine_index[16,:] = [8, 5]
            self.mine_index[17,:] = [8, 2]
            self.mine_index[18,:] = [9, 1]
            self.mine_index[19,:] = [6, 1]
            self.mine_index[20,:] = [4, 1]
            self.mine_index[21,:] = [4, 2]
            self.mine_index[22,:] = [7, 6]
            self.mine_index[23,:] = [4, 9]
            self.mine_index[24,:] = [4, 8]
            self.mine_index[25,:] = [6, 9]
        self.mine_pos = np.zeros(mine_num)
        for i in range(mine_num):
            self.mine_pos[i] = self.cal_pos(self.mine_index[i],self.MAX_X)

        # mapdata : inital_pos, terminal_pos, mine_pos  
        ######################################################
        ######################################################
        
    def __init__(self):
        self.obs = Observation()
        self.reset()

    def reset(self):
        self.make_MAP("fix")
        self.state = self.obs.reset(self.inital_pos, self.terminal_pos, self.mine_pos, self.MAX_X, self.MAX_Y)
        self.nA = 4
        self.npos = self.MAX_X * self.MAX_Y
        self.pos = self.inital_pos # 현재 위치

    def make_prob(self):
        P = {} # make next state
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=["multi_index"])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            

            P[s] = {a: [] for a in range(nA)} # 각 position state별로 가능한 action 수 (4개) 공간 만들기
            
            def is_done(state):
                if state in Terminal_state:
                    return True
                else:
                    return False
            
            ## reward shaping#################################################
            ##################################################################
            # 0번째공간 : up, 1번째공간 : right, 2번째공간 : down, 3번째공간 : left
            reward_label = [-1,-1,-1,-1] # reward 0:terminate -1:moving -2:hit the wall
            ##################################################################
            ##################################################################

            if y == 0 : # 벽이나 낭떠러지(바깥쪽) 이동하려 할 시
                ns_up_1 = s
                reward_label[0] = -2
            elif (s - self.MAX_X in mine_state) :
                ns_up_1 = s
                reward_label[0] = -2
            else : 
                ns_up_1 = s - self.MAX_X
            if x == (self.MAX_X - 1) :
                ns_right_1 = s
                reward_label[1] = -2
            elif (s + 1 in mine_state) :
                ns_right_1 = s
                reward_label[1] = -2 
            else : 
                ns_right_1 = s + 1
            if y == (self.MAX_Y - 1) :
                ns_down_1 = s
                reward_label[2] = -2
            elif (s + self.MAX_X in mine_state) :
                ns_down_1 = s
                reward_label[2] = -2
            else :
                ns_down_1 = s + self.MAX_X
            if x == 0 :
                ns_left_1 = s
                reward_label[3] = -2
            elif (s - 1 in mine_state) :
                ns_left_1 = s
                reward_label[3] = -2
            else : 
                ns_left_1 = s - 1
              
            P[s][UP] = [1, ns_up_1, reward_label, is_done(ns_up_1)]
            P[s][RIGHT] = [1, ns_right_1, reward_label, is_done(ns_right_1)]
            P[s][DOWN] = [1, ns_down_1, reward_label, is_done(ns_down_1)]
            P[s][LEFT] = [1, ns_left_1, reward_label, is_done(ns_left_1)]

            it.iternext()

