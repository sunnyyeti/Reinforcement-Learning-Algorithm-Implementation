import numpy as np
import seaborn
import matplotlib.pyplot as plt


class GridWorld():
    __EAST = "__east__"
    __WEST = "__west__"
    __SOUTH = "__south__"
    __NORTH = "__north__"
    ACTIONS = [__EAST, __WEST, __SOUTH, __NORTH]
    ROWS = 4
    COLS = 4
    TERMINATES = [(0,0),(ROWS-1,COLS-1)]

    def __init__(self):
        self.gridworld = np.zeros((GridWorld.ROWS, GridWorld.COLS), dtype=np.float32)

    def reset(self):
        self.gridworld = np.zeros((GridWorld.ROWS, GridWorld.COLS), dtype=np.float32)

    def __take_step(self, cell, direction):
        r, c = cell
        if direction == GridWorld.__EAST:
            if c == GridWorld.COLS - 1:
                return (r, c), -1
            return (r, c + 1), -1
        if direction == GridWorld.__WEST:
            if c == 0:
                return (r, c), -1
            return (r, c - 1), -1
        if direction == GridWorld.__SOUTH:
            if r == GridWorld.ROWS - 1:
                return (r, c), -1
            return (r + 1, c), -1
        if direction == GridWorld.__NORTH:
            if r == 0:
                return (r, c), -1
            return (r - 1, c), -1

    def solve(self, inplace=True, discount=1.0, iterations=100000, epsilon=1e-5):
        """
        :param inplace: whether to calculate the state values in place or not
        :param discount: reward discount for further steps.
        :param iterations: maximum number of iterations to solve the grid world problem.
        :param epsilon: calculating stops if difference between two iterations is smaller than epsilon for each grid cell.
        :return: None
        """
        for _ in range(iterations):
            source = self.gridworld if inplace else self.gridworld.copy()
            delta = 0
            for row in range(GridWorld.ROWS):
                for col in range(GridWorld.COLS):
                    if (row,col) not in GridWorld.TERMINATES:
                        orig_value = self.gridworld[row,col]
                        new_value = 0
                        for action in GridWorld.ACTIONS:
                            next_cell, reward = self.__take_step((row, col), action)
                            new_value += 0.25 * (reward + discount * source[next_cell])
                        self.gridworld[row,col] = new_value
                        delta = max(delta,abs(new_value-orig_value))
            if delta<epsilon:
                return


    def save_result(self, file_name):
        heatmap = seaborn.heatmap(self.gridworld, annot=True, fmt=".1f", cbar=False)
        figure = heatmap.get_figure()
        figure.savefig(file_name)
        plt.close()


def figure_4_1():
    grid = GridWorld()
    grid.solve(inplace=True, discount=1.0, iterations=1000000, epsilon=1e-5)
    grid.save_result("Figure_4_1.png")




if __name__ == "__main__":
    figure_4_1()

