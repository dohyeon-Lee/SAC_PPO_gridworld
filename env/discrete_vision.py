from gym import Env, spaces
from gym.utils import seeding
from gym.envs.toy_text.utils import categorical_sample
import numpy as np

class DiscreteEnv(Env):

    """
    Has the following members
    - nS: number of states
    - nA: number of actions
    - P: transitions (*)
    - isd: initial state distribution (**)

    (*) dictionary of lists, where
      P[s][a] == [(probability, nextstate, reward, done), ...]
    (**) list or array of length nS


    """
    def cal_index(self, s):
        x = s % self.MAX_X
        y = int(s / self.MAX_X)
        return x, y
    def cal_goal_direction(self, s, terminal_s):
        x_current, y_current = self.cal_index(s)
        x_terminal, y_terminal = self.cal_index(terminal_s)
        goal_dir = np.arctan2((y_terminal - y_current) ,(x_terminal - x_current))
        if self.lastaction == 0 : #up
            goal_direction = goal_dir + np.pi/2
        elif self.lastaction == 1 : #right
            goal_direction = goal_dir
        elif self.lastaction == 2 : #down
            goal_direction = goal_dir - np.pi/2
        else : # left
            goal_direction = goal_dir - np.pi
        if np.abs(goal_direction) > np.pi:
            if goal_direction > 0:
                goal_direction = goal_direction - 2*np.pi
            else :
                goal_direction = goal_direction + 2*np.pi
        return goal_direction


        
    def __init__(self, nS, nA, P, isd, MAX_X, MAX_Y, mine_state, Terminal_state):
        self.collision = 0
        self.P = P
        self.isd = isd
        self.lastaction = 1 # None  # for rendering
        self.nS = nS
        self.nA = nA
        self.MAX_X = MAX_X
        self.MAX_Y = MAX_Y
        self.mine_state = mine_state
        self.Ternimal_state = Terminal_state
        self.action_space = spaces.Discrete(self.nA)        # 큰 의미없음
        self.observation_space = spaces.Discrete(self.nS)   # 큰 의미없음
        self.move_count = 0 # 누적 이동횟수
        self.seed()
        #self.s = categorical_sample(self.isd, self.np_random)
        s_pos = categorical_sample(self.isd, self.np_random)
        self.goal_direction = self.cal_goal_direction(s_pos, self.Ternimal_state)
        self.state = np.array([self.lastaction, self.goal_direction])

        self.s = np.append(self.state, self.vision(self.lastaction, s_pos))
        self.s = np.append(self.s,s_pos)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #self.s = categorical_sample(self.isd, self.np_random)
        self.move_count = 0
        self.s = np.append(self.state, self.vision(self.lastaction, categorical_sample(self.isd, self.np_random)))
        self.s = np.append(self.s,categorical_sample(self.isd, self.np_random))
        self.lastaction = 1
        return self.s
    
    def vision(self, lastaction, s):
        #############################################
        # vision state :                            #
        #                   [4]                     #
        #                   [1]                     #
        #             [3][0][s][2][5]               #
        #            before action : up             #

        #                   [3]                     #
        #                   [0]                     #
        #                   [s][1][4]               #
        #                   [2]                     #
        #                   [5]                     #
        #            before action : right          #

        #                   [5]                     #
        #                   [2]                     #
        #             [4][1][s]                     #
        #                   [0]                     #
        #                   [3]                     #
        #            before action : left           #

        #             [5][2][s][0][3]               #
        #                   [1]                     #
        #                   [4]                     #
        #            before action : down           #
        #                                           #
        #############################################                        
        vision_state = np.zeros(6)
        x, y = self.cal_index(s)
        if lastaction == 0: # up
            
            if x == 0:                                  # 0
                vision_state[0] = 1
            elif (s - 1 in self.mine_state):
                vision_state[0] = 1   
            if y == 0:                                  # 1
                vision_state[1] = 1
            elif (s - self.MAX_X in self.mine_state): 
                vision_state[1] = 1
            if x == self.MAX_X - 1:                     # 2
                vision_state[2] = 1
            elif (s + 1 in self.mine_state): 
                vision_state[2] = 1
            
            if vision_state[0] == 1:                    # 3
                vision_state[3] == 1
            else:                       
                if x == 1:
                    vision_state[3] == 1
                elif (s - 2 in self.mine_state):
                    vision_state[3] == 1
            if vision_state[1] == 1:                    # 4
                vision_state[4] == 1
            else:
                if y == 1:
                    vision_state[4] == 1
                elif (s - 2*self.MAX_X in self.mine_state) :
                    vision_state[4] == 1
            if vision_state[2] == 1:                    # 5
                vision_state[5] == 1
            else:
                if x == self.MAX_X - 2:
                    vision_state[5] == 1
                elif (s + 2 in self.mine_state) :
                    vision_state[5] == 1
        
        elif lastaction == 1: # right
            
            if y == 0:                                  # 0
                vision_state[0] = 1
            elif (s - self.MAX_X in self.mine_state): 
                vision_state[0] = 1   
            if x == self.MAX_X - 1:                     # 1
                vision_state[1] = 1
            elif (s + 1 in self.mine_state): 
                vision_state[1] = 1
            if y == (self.MAX_Y - 1):                   # 2
                vision_state[2] = 1
            elif (s + self.MAX_X in self.mine_state): 
                vision_state[2] = 1
            
            if vision_state[0] == 1:                    # 3
                vision_state[3] == 1
            else:
                if y == 1:
                    vision_state[3] == 1
                elif (s - 2*self.MAX_X in self.mine_state) :
                    vision_state[3] == 1
            if vision_state[1] == 1:                    # 4
                vision_state[4] == 1
            else:
                if x == self.MAX_X - 2:
                    vision_state[4] == 1
                elif (s + 2 in self.mine_state) :
                    vision_state[4] == 1
            if vision_state[2] == 1:                    # 5
                vision_state[5] == 1
            else:
                if y == (self.MAX_Y - 2):
                    vision_state[5] == 1
                elif (s + 2*self.MAX_X in self.mine_state) :
                    vision_state[5] == 1
        
        elif lastaction == 3: # left
            
            if y == (self.MAX_Y - 1):                   # 0
                vision_state[0] = 1
            elif (s + self.MAX_X in self.mine_state): 
                vision_state[0] = 1
            if x == 0:                                  # 1
                vision_state[1] = 1
            elif (s - 1 in self.mine_state):
                vision_state[1] = 1 
            if y == 0:                                  # 2
                vision_state[2] = 1
            elif (s - self.MAX_X in self.mine_state): 
                vision_state[2] = 1 
            
            if vision_state[0] == 1:                    # 3
                vision_state[5] == 1
            else:
                if y == (self.MAX_Y - 2):
                    vision_state[3] == 1
                elif (s + 2*self.MAX_X in self.mine_state) :
                    vision_state[3] == 1
            if vision_state[1] == 1:                    # 4
                vision_state[4] == 1
            else:                       
                if x == 1:
                    vision_state[4] == 1
                elif (s - 2 in self.mine_state):
                    vision_state[4] == 1
            if vision_state[2] == 1:                    # 5
                vision_state[5] == 1
            else:
                if y == 1:
                    vision_state[5] == 1
                elif (s - 2*self.MAX_X in self.mine_state) :
                    vision_state[5] == 1
        
        elif lastaction == 2: # down
            
            if x == 0:                                  # 2
                vision_state[2] = 1
            elif (s - 1 in self.mine_state):
                vision_state[2] = 1   
            if y == (self.MAX_Y - 1):                   # 1
                vision_state[1] = 1
            elif (s + self.MAX_X in self.mine_state): 
                vision_state[1] = 1
            if x == self.MAX_X - 1:                     # 0
                vision_state[0] = 1
            elif (s + 1 in self.mine_state): 
                vision_state[0] = 1
            
            if vision_state[2] == 1:                    # 5
                vision_state[5] == 1
            else:                       
                if x == 1:
                    vision_state[5] == 1
                elif (s - 2 in self.mine_state):
                    vision_state[5] == 1
            if vision_state[1] == 1:                    # 4
                vision_state[4] == 1
            else:
                if y == (self.MAX_Y - 2):
                    vision_state[4] == 1
                elif (s + 2*self.MAX_X in self.mine_state) :
                    vision_state[4] == 1
            if vision_state[0] == 1:                    # 3
                vision_state[3] == 1
            else:
                if x == self.MAX_X - 2:
                    vision_state[3] == 1
                elif (s + 2 in self.mine_state) :
                    vision_state[3] == 1
        return vision_state

    def step(self, a):
        transitions = self.P[self.s[-1]][a] # action 받기 이전의 state에서, action을 환경에 넣음 -> next state와 reward를 줌
        #i = categorical_sample([t[0] for t in transitions], self.np_random) # transition : [(prob1, s_prime, reward1, is_done), (prob2, s_prime, reward2, is_done)], prob1,prob2확률대로 둘 중 하나 sampling
        #p, s, reward_label, d = transitions[i]
        p, s, reward_label, d = transitions
        self.lastaction = a
        ## calculate state ######################################
        self.goal_direction = self.cal_goal_direction(s, self.Ternimal_state)
        self.state = np.array([self.lastaction, self.goal_direction])
        
        self.s = np.append(self.state, self.vision(self.lastaction, s))
        self.s = np.append(self.s, s)


        ### 1000~0    100000/(1+collision**2)
        ## reward shaping #######################################
        if d == True : 
            reward_label[a] = 0
        if reward_label[a] == -1:
            r = -0.1*(self.goal_direction)**2
            # print("moving : {}".format(r))
        elif reward_label[a] == -2:
            r = -20
            self.collision += 1
            #print("hit the wall : {}".format(r))
        elif reward_label[a] == 0:
            r = 1000*(self.nS/self.move_count)
            r += 100000/(0.01+self.collision)
            # if self.collision < 100 and self.collision >= 50:
            #     r += 1000
            # elif self.collision < 50 and self.collision >= 10:
            #     r += 5000
            # elif self.collision < 10 and self.collision >= 1:
            #     r += 10000
            # elif self.collision < 1:
            #     r += 100000
            
            print("self collision : {}, END : {}".format(self.collision, r))
            self.collision = 0
        ## obervation except state ##############################
        self.move_count += 1

        return (self.s, r, d, {"prob": p}) # self.s : 10개 state
