import numpy as np

class MapGenerator():
    def __init__(self, MAX_X, MAX_Y):
        self.MAX_X = MAX_X
        self.MAX_Y = MAX_Y
        self.mine_num = 6
    
    def generate_mine_v1(self, kind = "small"):
        if kind == "small" : 
            self.mine_size = [3,3]
        retry = 1
        while retry == 1 : 
            block_num = 0
            generate_block_fail_num = 0
            # # generate first mine
            count = 0
            for i in [[0,0],[0,self.MAX_Y-(int(self.mine_size[1]/2) + 2)],[self.MAX_X- (int(self.mine_size[0]/2) + 2), self.MAX_Y- (int(self.mine_size[1]/2) + 2)],[self.MAX_X- (int(self.mine_size[0]/2) + 2), 0]]:
                count += 1
                mine_x = np.linspace(i[0], i[0] + self.mine_size[0] - 1, self.mine_size[0])
                mine_y = np.linspace(i[1], i[1] + self.mine_size[1] - 1, self.mine_size[1])
            #mine = [[0,0],[0,self.MAX_Y],[self.MAX_X, self.MAX_Y],[self.MAX_X, 0]]
                mine = []
                for x in mine_x:
                    for y in mine_y:
                        mine.append([x, y])
                if count == 1:
                    mine_list = mine
                    self.mine_batch = [mine]
                else : 
                    mine_list = np.append(mine_list, mine, axis=0)
                    self.mine_batch = np.append(self.mine_batch, [mine], axis=0)
            # input mine's info in mine_list and mine_batch
            while True : 
                mine_x_start = np.random.randint(0, self.MAX_X - (int(self.mine_size[0]/2) + 1))
                mine_y_start = np.random.randint(0, self.MAX_Y - (int(self.mine_size[1]/2) + 1))

                mine_x = np.linspace(mine_x_start, mine_x_start + self.mine_size[0] - 1, self.mine_size[0])
                mine_y = np.linspace(mine_y_start, mine_y_start + self.mine_size[1] - 1, self.mine_size[1])
                mine = []
                # print("==============")
                # print(mine_x)
                # print(mine_y)
                
                for x in mine_x:
                    for y in mine_y:
                        mine.append([x, y])
                # print(mine)
                # print("==============")
                #print("mine : {}".format(mine))
                #print("mine list : {}".format(mine_list))
                
                if any(np.all(np.equal(a_elem, b_elem)) for a_elem in mine for b_elem in mine_list):
                    #print("can't add..") 
                    generate_block_fail_num += 1  
                else : 
                    mine_list = np.append(mine_list, mine, axis=0)
                    self.mine_batch = np.append(self.mine_batch, [mine], axis=0)
                    block_num += 1
                if block_num >= self.mine_num:
                    retry = 0
                    break
                if generate_block_fail_num + block_num > 500:
                    retry = 1
                    break
                #print("now mine list : {}".format(mine_list))
                #print("batch : {}".format(self.mine_batch))
        self.mine_batch = self.mine_batch[~np.all(self.mine_batch == self.mine_batch[0], axis=(1, 2))]
        self.mine_batch = self.mine_batch[~np.all(self.mine_batch == self.mine_batch[0], axis=(1, 2))]
        self.mine_batch = self.mine_batch[~np.all(self.mine_batch == self.mine_batch[0], axis=(1, 2))]
        self.mine_batch = self.mine_batch[~np.all(self.mine_batch == self.mine_batch[0], axis=(1, 2))]
    
    def generate_mine_v2(self, minegrid_size = 3, hardpercent = 100, maxnum = 4):
        # 0 1 2 | 3 4 5 | 6 7 8 | 9 10 11
        minegrid_num_x = int(self.MAX_X / minegrid_size)
        minegrid_num_y = int(self.MAX_Y / minegrid_size)

        self.mine = []
        count = 0
        for i in range(minegrid_num_x):
            for j in range(minegrid_num_y):
                if (i == 0 or i == minegrid_num_x-1) and (j == 0 or j == minegrid_num_y-1):
                    continue
                mine_x = np.linspace(i * minegrid_size, (i + 1) * minegrid_size - 1, minegrid_size)
                mine_y = np.linspace(j * minegrid_size, (j + 1) * minegrid_size - 1, minegrid_size)
                #print(mine_x)
                if 0 == np.random.randint(0,2):
                    mine_x = np.delete(mine_x,0)
                else:
                    mine_x = np.delete(mine_x,minegrid_size-1)
                #print(mine_x)
                if 0 == np.random.randint(0,2):
                    mine_y = np.delete(mine_y,0)
                else:
                    mine_y = np.delete(mine_y,minegrid_size-1)
               
                if hardpercent >= np.random.randint(0,101):
                    count += 1
                    if count > maxnum:
                        continue
                    for x in mine_x:
                        for y in mine_y:
                            self.mine.append([x, y])

    def cal_pos(self, index):
        return (index[1] * self.MAX_X) + index[0]
    
    def generate_mine_pos_v1(self, mine_num):
        self.mine_num = mine_num
        self.generate_mine_v1("small")
        
        mine_numsize = int(self.mine_size[0]*self.mine_size[1])

        self.mine_pos_batch = []
        for mine in self.mine_batch:
            for i in range(mine_numsize):
                if (i >= self.mine_size[0]) and (i % self.mine_size[0] != 0):
                    #if ((i+1) % self.mine_size[0] != 0) and (i < (mine_numsize-self.mine_size[0])) :
                    mine_pos = self.cal_pos(mine[i])
                    self.mine_pos_batch = np.append(self.mine_pos_batch, mine_pos)
                
        return self.mine_pos_batch
    
    def generate_mine_pos_v2(self, minegrid_size=3, hardpercent=100, maxnum = 4):
        self.generate_mine_v2(minegrid_size, hardpercent, maxnum)

        self.mine_pos_batch = []
        for mine in self.mine:
            mine_pos = self.cal_pos(mine)
            self.mine_pos_batch = np.append(self.mine_pos_batch, mine_pos)
                
        return self.mine_pos_batch


# import numpy as np

# a = [[1, 2], [2, 2]]
# b = [[3, 3], [4, 4], [1, 1]]

# # 배열 a와 b의 요소들 중에서 서로 같은 값이 있는지 확인
# if any(np.all(np.equal(a_elem, b_elem)) for a_elem in a for b_elem in b):
#     print("a와 b의 요소들 중에 서로 같은 값이 있습니다.")
# else:
#     print("a와 b의 요소들은 모두 다릅니다.")

        
# gen = MapGenerator(30,30)
# gen.generate_mine_v2(minegrid_size = 3, hardpercent = 100)