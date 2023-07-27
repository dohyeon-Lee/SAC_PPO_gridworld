from gym import Env
from gym.utils import seeding
import numpy as np
import io
import sys, os
import time
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("env"))))
from env import gridworld_observation
from env import map_generator
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
        self.stepcount = 0
        return self.state
    
    def step(self, a): # imput : action output : next state, reward
        self.stepcount += 1
        next_pos, agent_condition, d = self.P[self.pos][a]
        self.pos = next_pos
        self.state, self.vision_pos = self.obs.step(self.pos, a)
        ################################################################
        #### make reward ###############################################
        reward = 0
        if d == True :  # terminate
            collision_reward = 100 - 0.1*self.collision
            if collision_reward < 0:
                collision_reward = 0
            reward = collision_reward
            #print(self.stepcount)
            self.stepcount = 0
            #print("# of hit wall : {}, self collision reward : {} ".format(self.collision, collision_reward))
            self.collision = 0
            self.move_count = 0
        elif self.stepcount > 10000 : 
            reward = - (self.state[1]**2)*0.1
            reward += -10
            self.stepcount = 0
            d = True
        elif agent_condition[a] == -1 : # move
            reward = - (self.state[1]**2)*0.1

        elif agent_condition[a] == -2 : # hit wall
            
            reward = - (self.state[1]**2)*0.1
            reward += -1
            self.collision += 1            
        self.move_count += 1

        return (self.state, reward/10, d)

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
        print("angle: {}".format(self.state[1]*(180/np.pi)))

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
            self.MAX_X = 9
            self.MAX_Y = 9
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
            
            self.mine_pos = np.zeros(mine_num)
            for i in range(mine_num):
                self.mine_pos[i] = self.cal_pos(self.mine_index[i],self.MAX_X)
        else : 
            map = map_generator.MapGenerator(self.MAX_X, self.MAX_Y)
            self.mine_pos = map.generate_mine_pos_v2(minegrid_size=3, hardpercent=-90, maxnum=0)

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
    for i in range(100):
        env.reset("fix")
        env._render()
        time.sleep(0.5)
    
if __name__ == "__main__":
    main()