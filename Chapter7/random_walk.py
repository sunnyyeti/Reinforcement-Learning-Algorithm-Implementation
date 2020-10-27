import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class RandomWalk:
    STATE_VALS = np.arange(-18, 20, 2) / 20.0
    TERMINATE_LEFT = -1
    TERMINATE_RIGHT = 19
    TERMINATES = {TERMINATE_LEFT, TERMINATE_RIGHT}
    START = 9

    def init(self):
        self.state_estimates = np.zeros_like(RandomWalk.STATE_VALS)

    def state_value(self, state):
        """
        :param state: current state
        :return: estimated current state value
        """
        if state in RandomWalk.TERMINATES:
            return 0.0
        return self.state_estimates[state]

    def update_state_value(self, state, val):
        self.state_estimates[state] = val

    def step(self, currstate):
        """
        state transition function
        :param currstate: current valid state, one of {0,1,2,3,4}
        :return: tuple, (reward, next_state)
        """
        next_state = currstate + np.random.choice([-1, 1])
        reward = 1 if next_state == RandomWalk.TERMINATE_RIGHT else -1 if next_state == RandomWalk.TERMINATE_LEFT else 0
        return (reward, next_state)

    def TD_n(self, n, alpha, gamma):
        T = float("inf")
        step = 0
        cur_state = RandomWalk.START
        states_trace = [cur_state]
        rewards_trace = [0]
        while True:
            if step < T:
                next_reward, next_state = self.step(cur_state)
                cur_state = next_state
                if next_state in RandomWalk.TERMINATES:
                    T = step + 1
                states_trace.append(next_state)
                rewards_trace.append(next_reward)
            update_step = step - n + 1
            if update_step >= 0:
                update_tar = 0.0
                for t in range(update_step + 1, min(T, update_step + n) + 1):
                    update_tar += gamma ** (t - update_step - 1) * rewards_trace[t]
                if update_step + n < T:
                    update_tar += gamma ** n * self.state_value(states_trace[update_step + n])
                update_state = states_trace[update_step]
                if update_state not in RandomWalk.TERMINATES:
                    update_to = self.state_value(update_state) + alpha * (update_tar - self.state_value(update_state))
                    self.update_state_value(update_state, update_to)
            if update_step == T - 1:
                break
            step += 1

    def rms_error(self):
        return np.sqrt(np.mean((RandomWalk.STATE_VALS - self.state_estimates) ** 2))

    def figure_7_2(self):
        gamma = 1.0
        alphas = np.arange(0.0, 1.1, 0.1)
        n_list = 2 ** np.arange(0, 10, 1)
        runs = 100
        episodes = 10
        lines = np.zeros((len(n_list), len(alphas)), dtype=np.float64)
        for row, steps in tqdm(enumerate(n_list)):
            for col, alpha in enumerate(alphas):
                for r in range(runs):
                    # print("RUN",r)
                    self.init()
                    for _ in range(episodes):
                        self.TD_n(steps, alpha, gamma)
                        lines[row, col] += self.rms_error()
        lines = lines / (runs * episodes)
        for i, line in enumerate(lines):
            plt.plot(alphas, line, label="n={}".format(n_list[i]))
        plt.legend()
        plt.grid()
        plt.xlabel("α")
        plt.ylabel("Average RMS error over 19 states and first 10 episodes")
        plt.ylim([0.25, 0.55])
        plt.savefig("Figure_7_2.png")
        plt.close()


def figure_7_2():
    random_walk = RandomWalk()
    random_walk.figure_7_2()


if __name__ == "__main__":
    figure_7_2()
