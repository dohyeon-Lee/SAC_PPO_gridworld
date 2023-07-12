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
    
    def __init__(self, max_x = 10, max_y = 10):
        
        self.MAX_Y = max_y
        self.MAX_X = max_x
        shape = [self.MAX_Y, self.MAX_X]
        self.shape = shape
        nS = np.prod(shape)
        nA = 4

        Inital_index = [0, 0]
        Terminal_index = [self.MAX_X-1, self.MAX_Y-1]
        Inital_state = self.cal_state(Inital_index, self.MAX_X)
        Terminal_state = self.cal_state(Terminal_index, self.MAX_X)
        
        #### MAP MAKING ######################################
        ######################################################
        mine_num = 0
        self.mine_index = np.zeros((mine_num,2))
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
        
        P = {}
        grid = np.arange(nS).reshape(shape)
        it = np.nditer(grid, flags=["multi_index"])
        
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index
            
            P[s] = {a: [] for a in range(nA)}
            
            def is_done(state):
                if state == Terminal_state:
                    return True
                else:
                    return False
            
            reward = 0.0 if is_done(s) else -1.0
            
            if is_done(s):
                P[s][UP] = [(0.5, s, reward, True), (0.5, s, reward, True)]
                P[s][RIGHT] = [(0.5, s, reward, True), (0.5, s, reward, True)]
                P[s][DOWN] = [(0.5, s, reward, True), (0.5, s, reward, True)]
                P[s][LEFT] = [(0.5, s, reward, True), (0.5, s, reward, True)]
            else:
                if s in mine_state:
                    ns_up_1 = Inital_state
                    ns_up_2 = Inital_state
                    ns_down_1 = Inital_state
                    ns_down_2 = Inital_state
                    ns_right_1 = Inital_state
                    ns_right_2 = Inital_state
                    ns_left_1 = Inital_state
                    ns_left_2 = Inital_state
                else:
                    # 한칸씩 움직이는 경우
                    ns_up_1 = s if y == 0 else s - self.MAX_X
                    ns_right_1 = s if x == (self.MAX_X - 1) else s + 1
                    ns_down_1 = s if y == (self.MAX_Y - 1) else s + self.MAX_X
                    ns_left_1 = s if x == 0 else s - 1
                    # 두칸씩 움직이는경우
                    if y == 0:
                        ns_up_2 = s
                    elif y == 1:
                        ns_up_2 = s - self.MAX_X
                    else:
                        ns_up_2 = s - self.MAX_X * 2

                    if x == (self.MAX_X - 1):
                        ns_right_2 = s
                    elif x == (self.MAX_X - 2):
                        ns_right_2 = s + 1
                    else:
                        ns_right_2 = s + 2

                    if y == (self.MAX_Y - 1):
                        ns_down_2 = s
                    elif y == (self.MAX_Y - 2):
                        ns_down_2 = s + self.MAX_X
                    else:
                        ns_down_2 = s + self.MAX_X * 2  

                    if x == 0:
                        ns_left_2 = s
                    elif x == 1:
                        ns_left_2 = s - 1
                    else:
                        ns_left_2 = s - 2
                prob = 0
                P[s][UP] = [(1 - prob / 100, ns_up_1, reward, is_done(ns_up_1)), (prob / 100, ns_up_2, reward, is_done(ns_up_2))]
                P[s][RIGHT] = [(1 - prob / 100, ns_right_1, reward, is_done(ns_right_1)), (prob / 100, ns_right_2, reward, is_done(ns_right_2))]
                P[s][DOWN] = [(1 - prob / 100, ns_down_1, reward, is_done(ns_down_1)), (prob / 100, ns_down_2, reward, is_done(ns_down_2))]
                P[s][LEFT] = [(1 - prob / 100, ns_left_1, reward, is_done(ns_left_1)), (prob / 100, ns_left_2, reward, is_done(ns_left_2))]
            it.iternext()
            
        # Initial state
        isd = np.zeros(nS)
        isd[Inital_state] = 1
        
        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P
        self.Inital_index = Inital_index
        self.Terminal_index = Terminal_index
        self.Inital_state = Inital_state
        self.Terminal_state = Terminal_state
        self.mine_num = mine_num
        self.mine_state = mine_state
        super(GridworldEnv, self).__init__(nS, nA, P, isd)

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

            if self.s == s:
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