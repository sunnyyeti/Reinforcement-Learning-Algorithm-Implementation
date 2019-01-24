import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class WindyGridWorld:
    """
    Some constant class variables
    """
    UP = "__up__"
    DOWN = "__down__"
    RIGHT = "__right__"
    LEFT = "__LEFT__"
    ACTIONS = [UP, DOWN, RIGHT, LEFT]
    WINDS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0], dtype=np.int8)
    START = (3, 0)
    END = (3, 7)
    ROWS = 7
    COLS = 10
    STEP_REWARD = -1.0

    def init(self):
        self.grid_word = np.zeros((WindyGridWorld.ROWS, WindyGridWorld.COLS, len(WindyGridWorld.ACTIONS)),
                                  dtype=np.float32)

    def epsilon_greedy(self, state,
                       epsilon):  # get an action index for given state according to epsilon-greeedy strategy
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(list(range(len(WindyGridWorld.ACTIONS))))
        else:
            max_action_value = np.max(self.grid_word[state])
            action_ind = np.random.choice(
                [i for i in range(len(WindyGridWorld.ACTIONS)) if self.grid_word[state][i] == max_action_value])
            return action_ind

    def step(self, state, action_ind):
        row, col = state
        action = WindyGridWorld.ACTIONS[action_ind]
        if action == WindyGridWorld.UP:
            next_row = max(0, row - 1 - WindyGridWorld.WINDS[col])
            next_col = col

        if action == WindyGridWorld.DOWN:
            next_row = min(WindyGridWorld.ROWS - 1, max(0, row + 1 - WindyGridWorld.WINDS[col]))
            next_col = col

        if action == WindyGridWorld.RIGHT:
            next_row = max(0, row - WindyGridWorld.WINDS[col])
            next_col = min(col + 1, WindyGridWorld.COLS - 1)

        if action == WindyGridWorld.LEFT:
            next_row = max(0, row - WindyGridWorld.WINDS[col])
            next_col = max(col - 1, 0)
        return next_row, next_col

    def go_one_episode(self, alpha, epsilon):
        state = WindyGridWorld.START
        action = self.epsilon_greedy(state, epsilon)
        steps = 0
        while state != WindyGridWorld.END:
            next_state = self.step(state, action)
            next_action = self.epsilon_greedy(next_state, epsilon)
            update = alpha * (WindyGridWorld.STEP_REWARD + self.grid_word[next_state[0], next_state[1], next_action] -
                              self.grid_word[state[0], state[1], action])
            self.grid_word[state[0], state[1], action] += update
            state = next_state
            action = next_action
            steps += 1
        return steps

    def run_episodes(self, episodes, alpha, epsilon):
        episode_steps = [0]
        for _ in tqdm(range(episodes)):
            episode_steps.append(self.go_one_episode(alpha, epsilon))
        return np.add.accumulate(episode_steps)

    def __draw_line(self, point1, point2, color):  # point is a tuple of coordinates (x,y)
        x, y = list(zip(point1, point2))
        plt.plot(x, y, color=color)

    def __draw_grid(self, top_left, cell_width, color):  # point is a tuple of coordinates (x,y)
        width = WindyGridWorld.COLS * cell_width
        height = WindyGridWorld.ROWS * cell_width
        self.__draw_line(top_left, (top_left[0] + width, top_left[1]), color)
        for i in range(1, WindyGridWorld.ROWS + 1):
            y = top_left[1] - i * cell_width
            self.__draw_line((top_left[0], y), (top_left[0] + width, y), color)
        self.__draw_line(top_left, (top_left[0], top_left[1] - height), color)
        for i in range(1, WindyGridWorld.COLS + 1):
            x = top_left[0] + i * cell_width
            self.__draw_line((x, top_left[1]), (x, top_left[1] - height), color)

    def __get_cell_centre(self, cell, top_left,
                          cell_width):  # get coordinates of the center in a cell indicated by (row,col)
        r, c = cell
        x, y = top_left
        centre_y = y - (r + 0.5) * cell_width
        centre_x = x + (c + 0.5) * cell_width
        return centre_x, centre_y

    def visualize_trajectory(self):
        TOP_LEFT = (10, 100)
        CELL_WIDTH = 5
        actions = self.grid_word.argmax(axis=-1)
        self.__draw_grid(TOP_LEFT, CELL_WIDTH, "r")
        state = WindyGridWorld.START
        action = actions[state]
        while state != WindyGridWorld.END:
            next_state = self.step(state, action)
            self.__draw_line(self.__get_cell_centre(state, TOP_LEFT, CELL_WIDTH),
                             self.__get_cell_centre(next_state, TOP_LEFT, CELL_WIDTH), "g")
            state = next_state
            action = actions[state]
        start_center = self.__get_cell_centre(WindyGridWorld.START, TOP_LEFT, CELL_WIDTH)
        end_center = self.__get_cell_centre(WindyGridWorld.END, TOP_LEFT, CELL_WIDTH)
        TEXT_COLOR = "b"
        TEXT_SIZE = "xx-large"
        plt.text(start_center[0], start_center[1], "S", color=TEXT_COLOR, size=TEXT_SIZE)
        plt.text(end_center[0], end_center[1], "G", color=TEXT_COLOR, size=TEXT_SIZE)
        for c_ in range(WindyGridWorld.COLS):
            R = WindyGridWorld.ROWS
            virtual_center_ = self.__get_cell_centre((R, c_), TOP_LEFT, CELL_WIDTH)
            plt.text(virtual_center_[0], virtual_center_[1], str(WindyGridWorld.WINDS[c_]), color=TEXT_COLOR,
                     size=TEXT_SIZE)
        plt.ylim([TOP_LEFT[1] - R * CELL_WIDTH - CELL_WIDTH, TOP_LEFT[1] + CELL_WIDTH])

    def example_6_5(self):
        EPISODES = 1000
        ALPHA = 0.5
        EPSILON = 0.1
        accumulated_steps = self.run_episodes(EPISODES, ALPHA, EPSILON)
        plt.figure(figsize=(40, 18))
        plt.subplot(1, 2, 1)
        plt.plot(accumulated_steps, np.arange(len(accumulated_steps)))
        plt.subplot(1, 2, 2)
        self.visualize_trajectory()
        plt.savefig("Example_6_5.png")
        plt.close()


def example_6_5():
    windy_gridword = WindyGridWorld()
    windy_gridword.init()
    windy_gridword.example_6_5()


if __name__ == "__main__":
    example_6_5()
