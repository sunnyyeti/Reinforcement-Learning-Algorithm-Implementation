import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Bandit():
    EPS_GREEDY = "__eps_greedy__"
    UCB = "__upper_confidence_bound__"
    GRADIENT = "__gradient_bandit__"

    def __init__(self, k_arms, q_true_mean, q_true_std, arm_reward_std, q_estimation_initial, strategy,
                 sample_average=True,
                 step_size=None):
        """
        :param K_arms: number of arms in a bandit problem
        :param q_true_mean: double. The mean of the normal distribution used to sample true action values for each arm.
        :param q_true_std: double. The standard deviation of the normal distribution used to sample true action values for each arm.
        :param arm_reward_std: double. The standard deviation used to sample the actual reward at each step for each arm.
        :param q_estimation_initial: double. The initial estimation values for each arm.
        :param strategy: dict. Key "name" is required, the value could be one of {Bandit.EPS_GREEDY, Bandit.UCB, Bandit.GRADIENT}
                               Other keys required:
                                For Bandit.EPS_GREEDY, key "epsilon:double" is required.
                                For Bandit.USB, key "coef_c:double" is required.
                                For Bandit.GRADIENT, key "alpha:double" and "ave_baseline:Bool" is required.
        :param sample_average: bool. Whether to update the action value estimations by sample averages. If True, step_size is ignored. This parameter is only used when strategy is not Bandit.GRADIENT
        :param step_size: double. Constant step_size for updating action value estimations. This parameter is only used when strategy is not Bandit.GRADIENT
        """
        self.k_arms = k_arms
        self.actions = np.arange(self.k_arms)
        self.q_true_mean = q_true_mean
        self.q_true_std = q_true_std
        self.arm_reward_std = arm_reward_std
        self.q_estimation_initial = q_estimation_initial
        self.strategy = strategy
        self.sample_average = sample_average
        self.step_size = step_size

    def init(self):
        """
        Initialize a new bandit problem.
        """
        self.q_true = self.q_true_std * np.random.randn(self.k_arms) + self.q_true_mean
        self.q_estimation = np.zeros(self.k_arms, dtype=np.float64) + self.q_estimation_initial
        self.action_count = np.zeros(self.k_arms, dtype=np.int32)
        self.best_action = np.argmax(self.q_true)
        self.step = 0
        self.ave_reward = 0

    def __random_choice_ind_over_max(self, arr):
        """
        Choice an index of the maximum values in an array randomly. Break the tie arbitrarily.
        :param arr: numpy array.
        :return: int.
        """
        max_est = np.max(arr)
        action = np.random.choice([act for act, q_est in enumerate(arr) if q_est == max_est])
        return action

    def __soft_max(self, arr):
        max_v = max(arr)
        return np.exp(arr - max_v) / np.sum(np.exp(arr - max_v))

    def select_action(self):
        """
        select an action according to the strategy
        :return: int.
        """
        if self.strategy["name"] == Bandit.EPS_GREEDY:
            epsilon = self.strategy["epsilon"]
            if np.random.rand() < epsilon:
                action = np.random.choice(self.actions)
                return action
            else:
                return self.__random_choice_ind_over_max(self.q_estimation)
        if self.strategy["name"] == Bandit.UCB:
            coef_c = self.strategy["coef_c"]
            q_ucb = self.q_estimation + coef_c * np.sqrt(np.log(self.step + 1) / (self.action_count + 1e-8))
            return self.__random_choice_ind_over_max(q_ucb)
        if self.strategy["name"] == Bandit.GRADIENT:
            q_probs = self.__soft_max(self.q_estimation)
            return np.random.choice(self.actions, p=q_probs)

    def take_step(self, action):
        self.step += 1
        reward = np.random.randn() * self.arm_reward_std + self.q_true[action]
        if self.strategy["name"] == Bandit.GRADIENT:
            if self.strategy["ave_baseline"]:
                self.ave_reward += (reward - self.ave_reward) / self.step
            baseline = self.ave_reward
            alpha = self.strategy["alpha"]
            one_hot_flag = np.zeros(self.k_arms)
            one_hot_flag[action] = 1
            q_probs = self.__soft_max(self.q_estimation)
            self.q_estimation += alpha * (reward - baseline) * (one_hot_flag - q_probs)
        else:
            self.action_count[action] += 1
            if self.sample_average:
                self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
            else:
                self.q_estimation[action] += (reward - self.q_estimation[action]) * self.step_size
        return reward

    def simulate(self, runs, steps):
        reward_matrix = np.zeros(steps, dtype=np.float64)
        action_matrix = np.zeros(steps, dtype=np.float64)
        for r in tqdm(range(1, runs + 1)):
            self.init()
            for s in range(steps):
                act = self.select_action()
                correct_act = (act == self.best_action)
                action_matrix[s] += (correct_act - action_matrix[s]) / r
                reward = self.take_step(act)
                reward_matrix[s] += (reward - reward_matrix[s]) / r
        return reward_matrix, action_matrix


def simulate(bandits, runs, steps):
    reward_maxtrix = np.zeros((len(bandits), steps), dtype=np.float64)
    action_matrix = np.zeros((len(bandits), steps), dtype=np.float64)
    for i, b in enumerate(bandits):
        b_reward, b_action = b.simulate(runs, steps)
        reward_maxtrix[i] = b_reward
        action_matrix[i] = b_action
    return reward_maxtrix, action_matrix


def figure_2_1():
    arms = 10
    data_num = 2000
    q_true = np.random.randn(arms)
    data = np.random.randn(data_num, arms) + q_true
    plt.violinplot(data, showmeans=True)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.savefig("Figure_2_1.png")
    plt.close()


def figure_2_2():
    epsilons = [0.0, 0.01, 0.1]
    bandit_par = {"k_arms": 10, "q_true_mean": 0, "q_true_std": 1, "arm_reward_std": 1, "q_estimation_initial": 0}
    strategies = ({"name": Bandit.EPS_GREEDY, "epsilon": eps} for eps in epsilons)
    bandits = []
    runs = 2000
    steps = 1000
    for strategy in strategies:
        bandit_par["strategy"] = strategy
        bandits.append(Bandit(**bandit_par))
    rewards, actions = simulate(bandits, runs, steps)
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    for eps, res in zip(epsilons, rewards):
        plt.plot(np.arange(1, steps + 1), res, label="ε = {:.2f}".format(eps))
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.subplot(212)
    for eps, act in zip(epsilons, actions):
        plt.plot(np.arange(1, steps + 1), act, label="ε = {:.2f}".format(eps))
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.savefig("Figure_2_2.png")
    plt.close()

def figure_2_3():
    common_ban_par = {"k_arms": 10, "q_true_mean": 0, "q_true_std": 1, "arm_reward_std": 1, "sample_average":False, "step_size":0.1}
    par1 = {"strategy":{"name":Bandit.EPS_GREEDY,"epsilon":0.0},"q_estimation_initial":5}
    par2 = {"strategy":{"name":Bandit.EPS_GREEDY,"epsilon":0.1},"q_estimation_initial":0}
    pars = [par1,par2]
    bandits =[]
    runs = 2000
    steps = 1000
    for par in pars:
        common_ban_par.update(par)
        bandits.append(Bandit(**common_ban_par))
    rewards, actions = simulate(bandits, runs, steps)
    labels = ["optimistic, greedy, Q1=5, ε=0", "realistic, ε-greedy, Q1=0, ε=0.1"]
    for act,lab in zip(actions,labels):
        plt.plot(np.arange(1, steps + 1), act, label=lab)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.savefig("Figure_2_3.png")
    plt.close()

def figure_2_4():
    common_ban_par = {"k_arms": 10, "q_true_mean": 0, "q_true_std": 1, "arm_reward_std": 1, "q_estimation_initial":0}
    par1 = {"strategy":{"name":Bandit.UCB,"coef_c":2}}
    par2 = {"strategy":{"name":Bandit.EPS_GREEDY,"epsilon":0.1}}
    pars = [par1,par2]
    bandits =[]
    runs = 2000
    steps = 1000
    for par in pars:
        common_ban_par.update(par)
        bandits.append(Bandit(**common_ban_par))
    rewards, actions = simulate(bandits, runs, steps)
    labels = ["UCB c=2", "ε-greedy ε=0.1"]
    for rew,lab in zip(rewards,labels):
        plt.plot(np.arange(1, steps + 1), rew, label=lab)
    plt.xlabel("Steps")
    plt.ylabel("Average reward")
    plt.legend()
    plt.savefig("Figure_2_4.png")
    plt.close()

def figure_2_5():
    common_ban_par = {"k_arms": 10, "q_true_mean": 4, "q_true_std": 1, "arm_reward_std": 1, "q_estimation_initial": 0}
    alphas = [0.1, 0.4]
    ave_baselines = [True, False]
    pars = [{"strategy":{"name":Bandit.GRADIENT,"alpha":alpha,"ave_baseline":ave_baseline}} for alpha in alphas for ave_baseline in ave_baselines]
    labels = ["α={:.1f}, with baseline={}".format(alpha,ave_baseline) for alpha in alphas for ave_baseline in ave_baselines]
    bandits = []
    runs = 2000
    steps = 1000
    for par in pars:
        common_ban_par.update(par)
        bandits.append(Bandit(**common_ban_par))
    rewards, actions = simulate(bandits, runs, steps)
    for act,lab in zip(actions,labels):
        plt.plot(np.arange(1, steps + 1), act, label=lab)
    plt.xlabel("Steps")
    plt.ylabel("% Optimal action")
    plt.legend()
    plt.savefig("Figure_2_5.png")
    plt.close()

def figure_2_6():
    epsilons = np.arange(-7,-1,dtype=np.float64)
    alphas = np.arange(-5,2,dtype=np.float64)
    coef_cs = np.arange(-4,3,dtype=np.float64)
    initials = np.arange(-2,3,dtype=np.float64)
    labels = ["ε-greedy","gradient bandit","UCB","greedy with optimistic initialization α=0.1"]
    pars = [epsilons,alphas,coef_cs,initials]
    common_ban_par = {"k_arms": 10, "q_true_mean": 0, "q_true_std": 1, "arm_reward_std": 1, "q_estimation_initial": 0}
    eps_par = lambda x: {"strategy":{"name":Bandit.EPS_GREEDY,"epsilon":x}}
    gra_par = lambda x: {"strategy":{"name":Bandit.GRADIENT,"alpha":x,"ave_baseline":True}}
    ucb_par = lambda x: {"strategy":{"name":Bandit.UCB,"coef_c":x}}
    ini_par = lambda x: {"strategy":{"name":Bandit.EPS_GREEDY,"epsilon":0},"q_estimation_initial":x,"sample_average":False,"step_size":0.1}
    par_gens = [eps_par,gra_par,ucb_par,ini_par]
    bandits = []
    for par_gen, par in zip(par_gens, pars):
        par = np.power(2,par)
        for p in par:
            common_ban_par.update(par_gen(p))
            bandits.append(Bandit(**common_ban_par))
    runs  = 2000
    steps = 1000
    rewards,actions = simulate(bandits,runs,steps)
    ave_reward_over_steps = rewards.mean(axis=1)
    #plt.figure(figsize=(10,10))
    indx=0
    for  par, lab in zip(pars, labels):
        x_ = np.power(2,par)
        y_ = ave_reward_over_steps[indx:indx+len(par)]
        indx = indx+len(par)
        plt.semilogx(x_,y_,label = lab,basex=2)
    plt.legend()
    plt.grid()
    plt.savefig("Figure_2_6.png")
    plt.close()


if __name__ == "__main__":
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()