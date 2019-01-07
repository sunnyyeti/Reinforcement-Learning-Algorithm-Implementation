import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn

class StateMachine:
    LEFT = "__left__"
    RIGHT = "__right__"
    START = "__start__"
    END = "__end__"

    def set_policy(self,policy):
        self.policy = policy

    def take_action(self,action):
        if action==StateMachine.RIGHT:
            return 0,StateMachine.END
        if action==StateMachine.LEFT:
            if np.random.binomial(1,0.9)==1:
                return 0,StateMachine.START
            else:
                return 1,StateMachine.END

    def run(self):
        trajectory = []
        state = StateMachine.START
        while state!=StateMachine.END:
            act = self.policy()
            trajectory.append(act)
            reward, state = self.take_action(act)
        return reward,trajectory

    def estimate_state_value(self,episodes,target_policy):
        state_values = np.zeros(episodes,dtype=np.float32)
        state_accured_scaled_reward = 0.0
        state_accured_scaled_weights = 1e-12
        for episode in tqdm(range(episodes)):
            reward,trajectory = self.run()
            scale = 1.0
            for act in trajectory:
                if act==target_policy():
                    scale*=2
                else:
                    scale = 0.0
                    break
            scaled_reward = scale*reward
            state_accured_scaled_reward+=scaled_reward
            state_accured_scaled_weights+=1.0
            state_values[episode] = state_accured_scaled_reward/state_accured_scaled_weights
        return state_values


def figure_5_4():
    sm = StateMachine()
    behaviour_policy = lambda : np.random.choice([StateMachine.RIGHT,StateMachine.LEFT])
    sm.set_policy(behaviour_policy)
    target_policy = lambda : StateMachine.LEFT
    runs = 10
    episodes = 1000000
    plt.figure(figsize=(10,10))
    for r in range(runs):
        plt.plot(np.arange(1,episodes+1),sm.estimate_state_value(episodes,target_policy))
    plt.xscale("log")
    plt.ylabel("Monte-carlo estimate of start value with ordinary importance sampling(ten runs)")
    plt.xlabel("Episodes (log scale)")
    plt.savefig("Figure_5_4.png")
    plt.close()

if __name__ == "__main__":
    figure_5_4()




