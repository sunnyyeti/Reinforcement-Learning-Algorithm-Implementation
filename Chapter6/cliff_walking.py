import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# import matplotlib.patches as patches


class CliffWalking:
    """
    Some constant class variables
    """
    ROWS = 4
    COLS = 12
    UP = "__up__"
    DOWN = "__down__"
    RIGHT = "__right__"
    LEFT = "__LEFT__"
    ACTIONS = [UP, DOWN, RIGHT, LEFT]
    START = (ROWS - 1, 0)
    GOAL = (ROWS - 1, COLS - 1)

    def init(self):
        self.grid_word = np.zeros((CliffWalking.ROWS, CliffWalking.COLS, len(CliffWalking.ACTIONS)),
                                  dtype=np.float32)

    def isCliff(self, row, col):
        return row == CliffWalking.ROWS - 1 and 1 <= col <= CliffWalking.COLS - 2

    def epsilon_greedy(self, state, epsilon):
        # get an action index for a given state according to epsilon-greeedy strategy
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(list(range(len(CliffWalking.ACTIONS))))
        else:
            max_action_value = np.max(self.grid_word[state])
            action_ind = np.random.choice(
                [i for i in range(len(CliffWalking.ACTIONS)) if self.grid_word[state][i] == max_action_value])
            return action_ind

    def step(self, state, action_ind):
        row, col = state
        action = CliffWalking.ACTIONS[action_ind]
        if action == CliffWalking.UP:
            next_row = max(0, row - 1)
            next_col = col

        if action == CliffWalking.DOWN:
            next_row = min(CliffWalking.ROWS - 1, row + 1)
            next_col = col

        if action == CliffWalking.RIGHT:
            next_row = row
            next_col = min(col + 1, CliffWalking.COLS - 1)

        if action == CliffWalking.LEFT:
            next_row = row
            next_col = max(col - 1, 0)

        if self.isCliff(next_row, next_col):
            return -100.0, CliffWalking.START
        else:
            return -1.0, (next_row, next_col)

    def sarsa_episode(self, alpha, epsilon):
        state = CliffWalking.START
        action = self.epsilon_greedy(state, epsilon)
        rewards = 0.0
        while state != CliffWalking.GOAL:
            reward, next_state = self.step(state, action)
            next_action = self.epsilon_greedy(next_state, epsilon)
            update = alpha * (reward + self.grid_word[next_state[0], next_state[1], next_action] -
                              self.grid_word[state[0], state[1], action])
            self.grid_word[state[0], state[1], action] += update
            state = next_state
            action = next_action
            rewards += reward
        return rewards

    def q_learning_episode(self, alpha, epsilon):
        state = CliffWalking.START
        action = self.epsilon_greedy(state, epsilon)
        rewards = 0.0
        while state != CliffWalking.GOAL:
            reward, next_state = self.step(state, action)
            update = alpha * (reward + self.grid_word[next_state].max() - self.grid_word[state[0], state[1], action])
            self.grid_word[state[0], state[1], action] += update
            state = next_state
            action = self.epsilon_greedy(state, epsilon)
            rewards += reward
        return rewards

    def expected_sarsa_episode(self, alpha, epsilon):
        state = CliffWalking.START
        action = self.epsilon_greedy(state, epsilon)
        rewards = 0.0
        while state != CliffWalking.GOAL:
            reward, next_state = self.step(state, action)
            next_state_max_q_value = self.grid_word[next_state].max()
            next_max_actions_cnt = sum(
                self.grid_word[next_state][i] == next_state_max_q_value for i in range(len(CliffWalking.ACTIONS)))
            expectded_q_value = 0.0
            for i in range(len(CliffWalking.ACTIONS)):
                if self.grid_word[next_state][i] == next_state_max_q_value:
                    expectded_q_value += next_state_max_q_value * (
                            epsilon / len(CliffWalking.ACTIONS) + (1 - epsilon) / next_max_actions_cnt)
                else:
                    expectded_q_value += self.grid_word[next_state][i] * (epsilon / len(CliffWalking.ACTIONS))
            update = alpha * (reward + expectded_q_value - self.grid_word[state][action])
            self.grid_word[state][action] += update
            state = next_state
            action = self.epsilon_greedy(state, epsilon)
            rewards += reward
        return rewards

    def do_experiments(self, runs, episodes, method, alpha, epsilon):
        res = 0.0
        for run_ind in range(runs):
            self.init()
            for epi_ind in range(episodes):
                res += method(alpha, epsilon)
        return res / (runs * episodes)

    def figure_6_3(self):
        EPSILON = 0.1
        ALPHAS = np.arange(0.1, 1.1, 0.1)
        METHOD_FUNCS = [self.sarsa_episode, self.q_learning_episode, self.expected_sarsa_episode]
        METHOD_NAMES = ["Sarsa", "Q-learning", "Expected Sarsa"]
        METHOD_MARKS = ["v", "s", "x"]
        METHOD_COLORS = ["blue", "black", "red"]
        PARS = [(500, 100, "Interim", ":"), (10, 1000, "Asymptotic", "-")]
        ## Because of the long running of initial setting
        ## For Interim, runs are reduced from 50000 to 500.
        ## For Asympotic epsidoes are reduced from 100000 to 1000.
        for method_func, method_name, method_mark, method_color in zip(METHOD_FUNCS, METHOD_NAMES, METHOD_MARKS,
                                                                       METHOD_COLORS):
            for runs, eppisodes, exp_name, line_style in PARS:
                res = np.zeros_like(ALPHAS)
                for i, alpha in enumerate(tqdm(ALPHAS)):
                    res[i] = self.do_experiments(runs, eppisodes, method_func, alpha, EPSILON)
                plt.plot(ALPHAS, res, linestyle=line_style, marker=method_mark, color=method_color,
                         label="{}_{}".format(method_name, exp_name))
        plt.legend()
        plt.savefig("Figure_6_3.png")
        plt.close()

    def __draw_line(self, ax, point1, point2, color, label=None):  # point is a tuple of coordinates (x,y)
        x, y = list(zip(point1, point2))
        ax.plot(x, y, color=color, label=label)

    def __draw_grid(self, ax, top_left, cell_width, color):  # point is a tuple of coordinates (x,y)
        width = CliffWalking.COLS * cell_width
        height = CliffWalking.ROWS * cell_width
        self.__draw_line(ax, top_left, (top_left[0] + width, top_left[1]), color)
        for i in range(1, CliffWalking.ROWS + 1):
            y = top_left[1] - i * cell_width
            self.__draw_line(ax, (top_left[0], y), (top_left[0] + width, y), color)
        self.__draw_line(ax, top_left, (top_left[0], top_left[1] - height), color)
        for i in range(1, CliffWalking.COLS + 1):
            x = top_left[0] + i * cell_width
            self.__draw_line(ax, (x, top_left[1]), (x, top_left[1] - height), color)

    def __get_cell_centre(self, cell, top_left, cell_width):
        # get coordinates of the center in a cell indicated by (row,col)
        r, c = cell
        x, y = top_left
        centre_y = y - (r + 0.5) * cell_width
        centre_x = x + (c + 0.5) * cell_width
        return centre_x, centre_y

    def __visualize_trajectory(self, ax, top_left, cell_width, color, label):
        TOP_LEFT = top_left
        CELL_WIDTH = cell_width
        actions = self.grid_word.argmax(axis=-1)
        state = CliffWalking.START
        action = actions[state]
        i = 0
        while state != CliffWalking.GOAL:
            _, next_state = self.step(state, action)
            self.__draw_line(ax, self.__get_cell_centre(state, TOP_LEFT, CELL_WIDTH),
                             self.__get_cell_centre(next_state, TOP_LEFT, CELL_WIDTH), color, label if not i else None)
            state = next_state
            action = actions[state]
            i += 1

    def __draw_background(self, ax, top_left, cell_width):
        TOP_LEFT = top_left
        CELL_WIDTH = cell_width
        self.__draw_grid(ax, TOP_LEFT, CELL_WIDTH, "black")
        start_center = self.__get_cell_centre(CliffWalking.START, TOP_LEFT, CELL_WIDTH)
        end_center = self.__get_cell_centre(CliffWalking.GOAL, TOP_LEFT, CELL_WIDTH)
        TEXT_COLOR = "black"
        TEXT_SIZE = "xx-large"
        ax.text(start_center[0], start_center[1], "S", color=TEXT_COLOR, size=TEXT_SIZE)
        ax.text(end_center[0], end_center[1], "G", color=TEXT_COLOR, size=TEXT_SIZE)
        # cliff_bottom_left_x = TOP_LEFT[0]+CELL_WIDTH
        # cliff_bottom_left_y = TOP_LEFT[1]-CELL_WIDTH*CliffWalking.ROWS
        # cliff_width = CELL_WIDTH*(CliffWalking.COLS-2)
        # cliff_height = CELL_WIDTH
        # patches.Rectangle((cliff_bottom_left_x,cliff_bottom_left_y),cliff_width,cliff_height,fill=True,color="gray")

    def example_6_6(self):
        RUNS = 50
        EPISODES = 1000
        ALPHA = 0.5
        EPSILON = 0.1
        TOP_LEFT = (10, 100)
        CELL_WIDTH = 5
        methods_funcs = [self.sarsa_episode, self.q_learning_episode]
        methods_names = ["Sarsa", "Q-learning"]
        methods_colors = ["red", "blue"]
        plt.figure(figsize=(20, 4))
        ax = plt.subplot(1, 2, 2)
        self.__draw_background(ax, TOP_LEFT, CELL_WIDTH)
        for func, name, color in zip(methods_funcs, methods_names, methods_colors):
            sum_rewards = np.zeros((RUNS, EPISODES), dtype=np.float32)
            for run in tqdm(range(RUNS)):
                self.init()
                for epi in range(EPISODES):
                    sum_rewards[run, epi] = func(ALPHA, EPSILON)
            # moving_average = np.copy(sum_rewards)
            # for i in range(9,len(moving_average)):
            #     moving_average[i] = np.mean(sum_rewards[i-9:i+1])
            ax = plt.subplot(1, 2, 1)
            ax.plot(np.arange(1, EPISODES + 1), sum_rewards.mean(axis=0), label=name, color=color)
            ax = plt.subplot(1, 2, 2)
            self.__visualize_trajectory(ax, TOP_LEFT, CELL_WIDTH, color, name)
        ax.legend()
        ax = plt.subplot(1, 2, 1)
        ax.legend()
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Sum of rewards during episodes")
        ax.set_ylim([-100, 0])
        plt.savefig("Example_6_6.png")
        plt.close()


def example_6_6():
    cliff_walk = CliffWalking()
    cliff_walk.example_6_6()


def figure_6_3():
    cliff_walk = CliffWalking()
    cliff_walk.figure_6_3()


if __name__ == "__main__":
    example_6_6()
    figure_6_3()
