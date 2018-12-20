import numpy as np

class Bandit():
    GREEDY = "__greedy__"
    EPS_GREEDY = "__eps_greedy__"
    UCB = "__upper_confidence_bound__"
    GRADIENT = "__gradient_bandit__"
    def __init__(self, K_arms, q_true_mean, q_true_var, arm_reward_var, q_estimation_initial, strategy, sample_average, step_size):
        """
        :param K_arms: number of arms in a bandit problem
        :param q_true_mean: double. The mean of the normal distribution used to sample true action values for each arm.
        :param q_true_var: double. The variance of the normal distribution used to sample true action values for each arm.
        :param arm_reward_var: double. The variance used to sample the actual reward at each step for each arm.
        :param q_estimation_initial: double. The initial estimation values for each arm.
        :param strategy: dict. Key "name" is required, the value could be one of {Bandit.GREEDY, Bandit.EPS_GREEDY, Bandit.UCB, Bandit.GRADIENT}
                               Other keys required:
                                For Bandit.GREEDY, no other keys are required.
                                For Bandit.EPS_GREEDY, key "epsilon" is required.
                                For Bandit.USB, key "coef_c" is required.
                                For Bandit.GRADIENT, key "alpha" is required.
        :param sample_average: bool. Whether to update the action value estimations by sample averages. If True, step_size is ignored.
        :param step_size: double. Constant step_size for updating action value estimations.
        """
        self.K_arms = K_arms
        self.q_true_mean = q_true_mean
        self.q_true_var = q_true_var
        self.arm_reward_var = arm_reward_var
        self.q_estimation_initial = q_estimation_initial
        self.strategy = strategy
        self.sample_average = sample_average
        self.step_size = step_size

    def init(self):
        """
        Initialize a new bandit problem.
        """
