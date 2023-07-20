import numpy as np

class Observation():

    def reset(self, inital_pos, terminal_pos, mine_pos, MAX_X, MAX_Y):
        self.MAX_X = MAX_X
        self.MAX_Y = MAX_Y
        self.mine_pos = mine_pos
        self.terminal_pos = terminal_pos
        self.inital_pos = inital_pos
        self.pos = inital_pos
        
        #######################################################
        #### reset state ######################################
        self.lastaction = 0
        self.compass = self.cal_goal_direction(self.pos)
        self.vision_state = self.vision(self.pos)
        state = [self.lastaction, self.compass]
        state = np.append(state, self.vision_state)
        self.state = state
        # state : [lastaction, compass, vision]
        return self.state

    def cal_goal_direction(self, s):
        x_current, y_current = self.cal_indexs(s)
        x_terminal, y_terminal = self.cal_indexs(self.terminal_pos)
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
    
    def vision(self, s):
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
        x, y = self.cal_indexs(s)
        if self.lastaction == 0: # up
            
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
        
        elif self.lastaction == 1: # right
            
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
        
        elif self.lastaction == 3: # left
            
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
        
        elif self.lastaction == 2: # down
            
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

