import numpy as np
import seaborn
import matplotlib.pyplot as plt
class GridWorld():
    RANDOM_POLICY = "__random_policy__"
    OPTIMAL_POLICY = "__optimal_policy__"
    __EAST = "__east__"
    __WEST = "__west__"
    __SOUTH = "__south__"
    __NORTH = "__north__"
    ACTIONS = [__EAST,__WEST,__SOUTH,__NORTH]
    ROWS = 5
    COLS = 5
    A_POS = (0,1)
    B_POS = (0,3)
    def __init__(self):
        self.gridworld = np.zeros((GridWorld.ROWS,GridWorld.COLS),dtype=np.float32)

    def reset(self):
        self.gridworld = np.zeros((GridWorld.ROWS,GridWorld.COLS),dtype=np.float32)

    def __take_step(self, cell, direction):
        if cell==GridWorld.A_POS:
            return (4,1),10
        if cell==GridWorld.B_POS:
            return (2,3),5
        r,c = cell
        if direction==GridWorld.__EAST:
            if c==GridWorld.COLS-1:
                return (r,c),-1
            return (r,c+1),0
        if direction==GridWorld.__WEST:
            if c==0:
                return (r,c),-1
            return(r,c-1),0
        if direction==GridWorld.__SOUTH:
            if r==GridWorld.ROWS-1:
                return (r,c),-1
            return (r+1,c),0
        if direction==GridWorld.__NORTH:
            if r==0:
                return (r,c), -1
            return (r-1,c),0

    def solve(self, policy, discount=0.9,iterations=100000, epsilon=1e-5):
        """
        :param policy: policy used to solve the problem. Allowed values:{GridWorld.RANDOM_POLICY, GridWorld.OPTIMAL_POLICY}.
        :param discount: reward discount for further steps.
        :param iterations: maximum number of iterations to solve the grid world problem.
        :param epsilon: calculating stops if difference between two iterations is smaller than epsilon for each grid cell.
        :return:
        """
        for _ in range(iterations):
            new_values = np.zeros_like(self.gridworld)
            for row in range(GridWorld.ROWS):
                for col in range(GridWorld.COLS):
                    if policy==GridWorld.RANDOM_POLICY:
                        for action in GridWorld.ACTIONS:
                            next_cell,reward = self.__take_step((row,col),action)
                            new_values[row,col] += 0.25*(reward+discount*self.gridworld[next_cell])
                    if policy==GridWorld.OPTIMAL_POLICY:
                        tmp_values = []
                        for action in GridWorld.ACTIONS:
                            next_cell,reward = self.__take_step((row,col),action)
                            tmp_values.append(reward+discount*self.gridworld[next_cell])
                        new_values[row,col] = max(tmp_values)
            diff = new_values-self.gridworld
            if np.all(np.abs(diff)<epsilon):
                self.gridworld = new_values
                return
            self.gridworld = new_values

    def save_result(self,file_name):
        heatmap = seaborn.heatmap(self.gridworld, annot=True, fmt=".1f", cbar=False)
        figure = heatmap.get_figure()
        figure.savefig(file_name)
        plt.close()


def figure_3_2():
    grid = GridWorld()
    grid.solve(policy=GridWorld.RANDOM_POLICY,discount=0.9,iterations=1000000,epsilon=1e-5)
    grid.save_result("Figure_3_2.png")

def figure_3_3():
    grid = GridWorld()
    grid.solve(policy=GridWorld.OPTIMAL_POLICY,discount=0.9,iterations=1000000,epsilon=1e-5)
    grid.save_result("Figure_3_3.png")

if __name__ == "__main__":
    figure_3_2()
    figure_3_3()