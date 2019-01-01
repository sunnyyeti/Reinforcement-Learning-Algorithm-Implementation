import numpy as np
import math
import seaborn
import matplotlib.pyplot as plt

class Gambler():
    TERMINATES = {0,100}
    WIN = 100
    def __init__(self,head_prob):
        self.reset(head_prob)

    def reset(self,head_prob):
        self.states = np.arange(0,101)
        self.state_values = np.zeros_like(self.states,dtype=np.float32)
        self.state_values[-1] = 1.0
        self.head_prob = head_prob

    def __possible_actions(self,state):
        end = min(state,Gambler.WIN-state)
        actions = np.arange(end+1)
        return actions

    def __step(self,state,action):
        next_state_head = state+action
        next_state_tail = state-action
        returns=self.head_prob*self.state_values[next_state_head]+(1-self.head_prob)*self.state_values[next_state_tail]
        return returns

    def value_iteration(self, eps, store_iterations):
        ite = 0
        cur_pos = 0
        cache = {}
        while True:
            max_delta = -1
            for state in self.states:
                if state not in Gambler.TERMINATES:
                    all_possible_actions = self.__possible_actions(state)
                    returns = [self.__step(state,action) for action in all_possible_actions]
                    max_return = max(returns)
                    max_delta = max(max_delta,abs(max_return-self.state_values[state]))
                    self.state_values[state]=max_return
            ite+=1
            if cur_pos<len(store_iterations) and ite==store_iterations[cur_pos]:
                cache[ite]=self.state_values.copy()
                cur_pos+=1
            #print("max_delta:{}",format(max_delta))
            if max_delta<eps:
                break
        return cache

    def extract_policy(self):
        self.policy = np.zeros_like(self.states)
        for state in self.states:
            if state not in Gambler.TERMINATES:
                all_possible_actions = self.__possible_actions(state)
                returns =  [self.__step(state,action) for action in all_possible_actions]
                max_action = all_possible_actions[np.argmax(np.round(returns[1:],5))+1] #最优policy有很多种，首先书本上的图里面是没有0的，
                #max_action = all_possible_actions[np.argmax(returns[1:]) + 1] #我们把它排除，其次数值解并不精确，精度只计算到小于1e-5而已,所以理论上相同的值需要四舍五入到同样的数值，这样才能选到最小。
                self.policy[state] = max_action

def figure_4_3():
    gambler = Gambler(0.4)
    store_iterations = [1,2,3,32]
    cached = gambler.value_iteration(eps=1e-5, store_iterations=store_iterations)
    plt.figure(figsize=(10,20))
    ax = plt.subplot(2,1,1)
    x_ticks = np.arange(1,100)
    for ite in sorted(list(cached.keys())):
        values = cached[ite][1:100]
        ax.plot(x_ticks,values,label="sweep {}".format(ite))
    ax.plot(x_ticks,gambler.state_values[1:100],label="Final value function")
    ax.set_xlabel("Capital")
    ax.set_ylabel("Value estimates")
    ax.legend()
    ax = plt.subplot(2,1,2)
    gambler.extract_policy()
    ax.plot(x_ticks,gambler.policy[1:100])
    ax.set_xlabel("Capital")
    ax.set_ylabel("Final policy")
    plt.savefig("Figure_4_3.png")
    plt.close()


if __name__ == "__main__":
    figure_4_3()




