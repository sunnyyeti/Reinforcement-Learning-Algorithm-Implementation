import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn


class RandomWalk:
    STATE_VALS = np.array([1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6], dtype=np.float32)
    TERMINATE_LEFT = -1
    TERMINATE_RIGHT = 5
    TERMINATES = {TERMINATE_LEFT, TERMINATE_RIGHT}

    def init(self):
        self.state_estimates = np.ones_like(RandomWalk.STATE_VALS) * 0.5

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
        reward = 1 if next_state == RandomWalk.TERMINATE_RIGHT else 0
        return (reward, next_state)

    def get_episode(self):
        """
        simulate a complete episode which must end at one terminal states
        :return: state list, reward list
        """
        states = []
        rewards = []
        next_state = 2
        while next_state not in RandomWalk.TERMINATES:
            states.append(next_state)
            curr, next_state = self.step(next_state)
            rewards.append(curr)
        states.append(next_state)
        rewards.append(0)
        return states, rewards

    def TD_0(self, alpha):
        states, rewards = self.get_episode()
        for i, (s, r) in enumerate(zip(states[:-1], rewards[:-1])):
            update = alpha * (r + self.state_value(states[i + 1]) - self.state_value(s))
            self.update_state_value(s, self.state_value(s) + update)

    def TD_0_batch(self, alpha, epsilon, batches):
        while True:
            updates = np.zeros_like(self.state_estimates)
            for states, rewards in batches:
                for i, (s, r) in enumerate(zip(states[:-1], rewards[:-1])):
                    updates[s] += alpha * (r + self.state_value(states[i + 1]) - self.state_value(s))
            self.state_estimates += updates
            if np.all(np.abs(updates) < epsilon):
                break

    def monte_carlo(self, alpha):
        states, rewards = self.get_episode()
        if states[-1] == RandomWalk.TERMINATE_LEFT:
            total_reward = 0
        else:
            total_reward = 1
        for s in states[:-1]:
            update = alpha * (total_reward - self.state_value(s))
            self.update_state_value(s, self.state_value(s) + update)

    def monte_carlo_batch(self, alpha, epsilon, batches):
        while True:
            updates = np.zeros_like(self.state_estimates)
            for states, rewards in batches:
                if states[-1] == RandomWalk.TERMINATE_LEFT:
                    total_reward = 0
                else:
                    total_reward = 1
                for s in states[:-1]:
                    updates[s] += alpha * (total_reward - self.state_value(s))
            self.state_estimates += updates
            if np.all(np.abs(updates) < epsilon):
                break

    def rms_error(self):
        return np.sqrt(np.mean((RandomWalk.STATE_VALS - self.state_estimates) ** 2))

    def example_6_2_left(self):
        self.init()
        myxticks = ["A", "B", "C", "D", "E"]
        plt.xticks(np.arange(5), myxticks)
        episodes = [0, 1, 10, 100, 10000]
        executed_episode = 0
        cur_target_ind = 0
        cur_target = episodes[cur_target_ind]
        while True:
            if executed_episode == cur_target:
                plt.plot(self.state_estimates, label="episode_{}".format(cur_target))
                cur_target_ind += 1
                if cur_target_ind == len(episodes):
                    break
                cur_target = episodes[cur_target_ind]
            self.TD_0(0.1)
            executed_episode += 1
        plt.plot(RandomWalk.STATE_VALS, label="true values")
        plt.legend()
        plt.grid()
        plt.xlabel("State")
        plt.title("Estimated value")

    def example_6_2_right(self):
        def experments(alpha, episodes, runs, method):
            if method == "TD":
                func = self.TD_0
            else:
                func = self.monte_carlo
            res = np.zeros((runs, episodes), dtype=np.float32)
            print("{}_{}".format(method, alpha))
            for r_ind in range(runs):
                self.init()
                for e_ind in range(episodes):
                    func(alpha)
                    res[r_ind][e_ind] = self.rms_error()
            return res.mean(axis=0)

        pars = {"TD": [0.05, 0.1, 0.15], "MC": [0.01, 0.02, 0.03, 0.04]}
        EPISODES = 100
        RUNS = 100
        for method, alphas in pars.items():
            linestyle = "solid" if method == "TD" else "dashdot"
            for alp in alphas:
                plt.plot(np.arange(1, EPISODES + 1), experments(alp, EPISODES, RUNS, method),
                         label="{}_{}".format(method, alp), linestyle=linestyle)
        plt.grid()
        plt.legend()
        plt.xlabel("Walks/Episodes")
        plt.title("Empirical RMS error, averaged over states")

    def exmaple_6_2(self):
        plt.figure(figsize=(40, 20))
        plt.subplot(1, 2, 1)
        self.example_6_2_left()
        plt.subplot(1, 2, 2)
        self.example_6_2_right()
        plt.savefig("Example_6_2.png")
        plt.close()

    def figure_6_2(self):
        ALPHA = 0.001
        EPSILON = 1e-3
        EPISODES = 100
        RUNS = 100
        METHODS = [self.TD_0_batch, self.monte_carlo_batch]
        METHOD_NAMES = ["TD", "MC"]
        plt.figure()
        for method_func, method_name in zip(METHODS, METHOD_NAMES):
            res = np.zeros((RUNS, EPISODES), dtype=np.float32)
            for run_ind in tqdm(range(RUNS)):
                self.init()
                batches = []
                for episode_ind in range(EPISODES):
                    batches.append(self.get_episode())
                    method_func(ALPHA, EPSILON, batches)
                    res[run_ind][episode_ind] = self.rms_error()
            plt.plot(np.arange(1, EPISODES + 1), res.mean(axis=0), label=method_name)
        plt.legend()
        plt.grid()
        plt.xlabel("Walks/Episodes")
        plt.ylabel("RMS error, averaged over states")
        plt.title("BATCH TRAINING")
        plt.savefig("Figure_6_2.png")
        plt.close()


def example_6_2():
    random_walk = RandomWalk()
    random_walk.exmaple_6_2()


def figure_6_2():
    random_walk = RandomWalk()
    random_walk.figure_6_2()


if __name__ == "__main__":
    # example_6_2()
    figure_6_2()
