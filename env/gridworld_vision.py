import io
import sys

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import discrete_vision
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class GridworldEnv(discrete_vision.DiscreteEnv):
    metadata = {'render.modes': ['human', 'ansi']}
    
    def cal_state(self, index, MAX_X):
        return (index[1] * MAX_X) + index[0]
    
    def mine_grid(self, mine_num, MAX_X, MAX_Y):
        x_list = np.random.randint(low=0, high=MAX_X, size=mine_num)
        y_list = np.random.randint(low=0, high=MAX_Y, size=mine_num)
        x_list = np.expand_dims(x_list, axis=0)
        y_list = np.expand_dims(y_list, axis=0)
        list = np.append(x_list, y_list, axis=0)
        list = list.T

        mine_list = list
        for i in range(mine_num):
            if(list[i,0] == 0 and list[i,1] == 0):
                mine_list = np.delete(list, i, axis = 0)
            elif(list[i,0] == MAX_X-1 and list[i,1] == MAX_Y-1):
                mine_list = np.delete(list, i, axis = 0)

        return mine_list
    def __init__(self, max_x = 10, max_y = 10):
        
        self.MAX_Y = max_y
        self.MAX_X = max_x
        shape = [self.MAX_Y, self.MAX_X]
        self.shape = shape
        nS = np.prod(shape) # num of grid position
        nA = 4

        Inital_index = [0, 0]
        Terminal_index = [self.MAX_X-1, self.MAX_Y-1]
        Inital_state = self.cal_state(Inital_index, self.MAX_X)
        Terminal_state = self.cal_state(Terminal_index, self.MAX_X)
        
        #### MAP MAKING ######################################
        ######################################################
        mine_num = 20
        self.mine_index = self.mine_grid(mine_num, self.MAX_X, self.MAX_Y)
        mine_num = self.mine_index.shape[0]
        # mine_num = 20
        # self.mine_index = np.zeros((mine_num,2))
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
        ######################################################
        ######################################################

        mine_state = np.zeros(mine_num)
        for i in range(mine_num):
            mine_state[i] = self.cal_state(self.mine_index[i],self.MAX_X)
        
        P = {} # make next state
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=["multi_index"])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            

            P[s] = {a: [] for a in range(nA)} # 각 position state별로 가능한 action 수 (4개) 공간 만들기
            
            def is_done(state):
                if state == Terminal_state:
                    return True
                else:
                    return False
            
            ## reward shaping#################################################
            ##################################################################
            reward = 0 if is_done(s) else -1 # reward 0:terminate -1:moving -2:hit the wall
            ##################################################################
            ##################################################################
            
            if is_done(s):
                P[s][UP] = [1, s, reward, True]
                P[s][RIGHT] = [1, s, reward, True]
                P[s][DOWN] = [1, s, reward, True]
                P[s][LEFT] = [1, s, reward, True]
            else:
                if y == 0 : # 벽이나 낭떠러지(바깥쪽) 이동하려 할 시
                    ns_up_1 = s
                    reward = -2
                elif (s - self.MAX_X in mine_state) :
                    ns_up_1 = s
                    reward = -2
                else : 
                    ns_up_1 = s - self.MAX_X
                if x == (self.MAX_X - 1) :
                    ns_right_1 = s
                    reward = -2
                elif (s + 1 in mine_state) :
                    ns_right_1 = s
                    reward = -2 
                else : 
                    ns_right_1 = s + 1
                if y == (self.MAX_Y - 1) :
                    ns_down_1 = s
                    reward = -2
                elif (s + self.MAX_X in mine_state) :
                    ns_down_1 = s
                    reward = -2
                else :
                    ns_down_1 = s + self.MAX_X
                if x == 0 :
                    ns_left_1 = s
                    reward = -2
                elif (s - 1 in mine_state) :
                    ns_left_1 = s
                    reward = -2
                else : 
                    ns_left_1 = s - 1
                    
                P[s][UP] = [1, ns_up_1, reward, is_done(ns_up_1)]
                P[s][RIGHT] = [1, ns_right_1, reward, is_done(ns_right_1)]
                P[s][DOWN] = [1, ns_down_1, reward, is_done(ns_down_1)]
                P[s][LEFT] = [1, ns_left_1, reward, is_done(ns_left_1)]

            it.iternext()
            
        # Initial state
        isd = np.zeros(nS)
        isd[Inital_state] = 1
        
        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.Inital_index = Inital_index
        self.Terminal_index = Terminal_index
        self.Inital_state = Inital_state
        self.Terminal_state = Terminal_state
        self.mine_num = mine_num
        self.mine_state = mine_state
        super(GridworldEnv, self).__init__(nS, nA, P, isd, self.MAX_X, self.MAX_Y, mine_state, Terminal_state)

    def _render(self, mode='human', close=False):
        """ Renders the current gridworld layout
         For example, a 4x4 grid with the mode="human" looks like:
            T  o  o  o
            o  x  o  o
            o  o  o  o
            o  o  o  T
        where x is your position and T are the two terminal states.
        """
        os.system('cls')
        if close:
            return

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s[-1] == s:
                output = "🔴"
            elif s == self.Terminal_state:
                output = "🟪"
            elif s == self.Inital_state:
                output = "🟩"
            elif s in self.mine_state:
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

def main():
    env = GridworldEnv()
    env._render()
    # for k in env.P.keys():

    #     print(f'env.P[{k}]={env.P[k]}')
    
if __name__ == "__main__":
    main()