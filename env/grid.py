import numpy as np
from numpy import random
def mine_grid(mine_num, MAX_X, MAX_Y):
    x_list = random.randint(low=0, high=MAX_X, size=mine_num)
    y_list = random.randint(low=0, high=MAX_Y, size=mine_num)
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

def main():
    list = mine_grid(20, 10, 10)
    print(list)

if __name__ == "__main__":
    main()
    