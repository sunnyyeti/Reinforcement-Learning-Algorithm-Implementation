import numpy as np
import math
import seaborn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class Poisson():
    CACHE = {}

    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, n):
        if (self.lambda_, n) in Poisson.CACHE:
            return Poisson.CACHE[(self.lambda_, n)]
        else:
            res = math.pow(self.lambda_, n) * math.exp(-self.lambda_) / math.factorial(n)
            Poisson.CACHE[(self.lambda_, n)] = res
            return res


class CarCompany():
    MAX_CARS = 20  # maximum number of cars at each location
    ACTIONS = np.arange(-5, 6,
                        dtype=np.int8)  # valid range of cars that can be moved from first location to second location
    POISSON_UPBOUND = 11  # upper bound for poisson distribution. Probability of getting number higher than this bound is 0.
    MOVE_COST = 2  # cost of moving one car from first location to second location
    RENTAL_FEE = 10  # credits for each car
    EXPECTED_REQUEST_1 = 3  # expected of cars requested at first location
    EXPECTED_REQUEST_2 = 4  # expected of cars requested at second location
    EXPECTED_RETURN_1 = 3  # expected of cars returned at first location
    EXPECTED_RETURN_2 = 2  # expected of cars returned at second location
    DISCOUNT = 0.9

    def __init__(self):
        self.state_values = np.zeros((CarCompany.MAX_CARS + 1, CarCompany.MAX_CARS + 1), dtype=np.float32)
        self.policy = np.zeros_like(self.state_values, dtype=np.int8)
        self.request1 = Poisson(CarCompany.EXPECTED_REQUEST_1)
        self.request2 = Poisson(CarCompany.EXPECTED_REQUEST_2)
        self.return1 = Poisson(CarCompany.EXPECTED_RETURN_1)
        self.return2 = Poisson(CarCompany.EXPECTED_RETURN_2)

    def __step(self, state, action):
        """
        :param state: tuple. Number of cars at each location at the end of the day.
        :param action: int. Number of cars moved from first location to second location.
        :return: float. Expected reward.
        """

        reward = -CarCompany.MOVE_COST * abs(action)
        for req1 in range(CarCompany.POISSON_UPBOUND + 1):
            first_loc = state[0]
            first_loc = min(first_loc - action, CarCompany.MAX_CARS)
            real_req1 = min(req1, first_loc)
            first_loc -= real_req1
            p1 = self.request1(req1)
            accp1 = p1
            for req2 in range(CarCompany.POISSON_UPBOUND + 1):
                second_loc = state[1]
                second_loc = min(second_loc + action, CarCompany.MAX_CARS)
                p2 = self.request2(req2)
                real_req2 = min(req2, second_loc)
                second_loc -= real_req2
                benefits = CarCompany.RENTAL_FEE * (real_req1 + real_req2)
                accp2 = accp1 * p2
                for ret1 in range(CarCompany.POISSON_UPBOUND + 1):
                    p3 = self.return1(ret1)
                    first_loc_ = min(first_loc + ret1, CarCompany.MAX_CARS)
                    accp3 = accp2 * p3
                    for ret2 in range(CarCompany.POISSON_UPBOUND + 1):
                        p4 = self.return2(ret2)
                        second_loc_ = min(second_loc + ret2, CarCompany.MAX_CARS)
                        prob = accp3 * p4
                        reward += prob * (benefits + CarCompany.DISCOUNT * self.state_values[first_loc_, second_loc_])
        return reward

    def policy_evaluation(self, epsilon=1e-4):
        step = 0
        while True:
            step += 1
            print(step)
            max_delta = -1
            for fir_loc in range(CarCompany.MAX_CARS + 1):
                for sec_loc in range(CarCompany.MAX_CARS + 1):
                    expected_return = self.__step((fir_loc, sec_loc), self.policy[fir_loc, sec_loc])
                    max_delta = max(max_delta, abs(expected_return - self.state_values[fir_loc, sec_loc]))
                    self.state_values[fir_loc, sec_loc] = expected_return
            print(max_delta)
            if max_delta < epsilon:
                break

    def policy_improvement(self):
        policy_stable = True
        for fir_loc in range(CarCompany.MAX_CARS + 1):
            for sec_loc in range(CarCompany.MAX_CARS + 1):
                q_values = []
                for action in CarCompany.ACTIONS:
                    if (action >= 0 and action <= fir_loc) or (action < 0 and (-action) <= sec_loc):
                        q_values.append(self.__step((fir_loc, sec_loc), action))
                    else:
                        q_values.append(float("-inf"))
                new_action = CarCompany.ACTIONS[np.argmax(q_values)]
                if new_action != self.policy[fir_loc, sec_loc]:
                    policy_stable = False
                    self.policy[fir_loc, sec_loc] = new_action
        return policy_stable

    def policy_iteration(self):
        while True:
            self.policy_evaluation()
            policy_stable = self.policy_improvement()
            if policy_stable:
                break


def figure_4_2():
    carcomp = CarCompany()
    plt.figure(figsize=(40, 20))
    ite = 1
    while True:
        ax = plt.subplot(2, 3, ite)
        fig = seaborn.heatmap(np.flipud(carcomp.policy),cmap="YlGnBu")
        fig.set_title("Policy_{}".format(ite))
        fig.set_xlabel("#Cars at second location")
        fig.set_ylabel("#Cars at first location")
        fig.set_yticks(list(reversed(range(CarCompany.MAX_CARS + 1))))
        print("evaluation...")
        carcomp.policy_evaluation()
        print("extraction...")
        policy_stable = carcomp.policy_improvement()
        print("stable", policy_stable)
        if policy_stable:
            ax = plt.subplot(2, 3, 6, projection="3d")
            x = np.arange(CarCompany.MAX_CARS + 1)
            y = np.arange(CarCompany.MAX_CARS + 1)
            x, y = np.meshgrid(x, y)
            ax.plot_surface(x, y, carcomp.state_values, cmap=cm.coolwarm)
            ax.set_xlabel("#Cars at second location")
            ax.set_ylabel("#Cars at first location")
            ax.set_title("Optimal state value")
            break
        ite += 1
    plt.savefig("Figure_4_2.png")
    plt.close()


if __name__ == "__main__":
    figure_4_2()
