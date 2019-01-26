import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class MDP:
    A_LEFT = 1
    A_RIGHT = 0
    A = 0
    B = 1
    END = 2
    ACTIONS = {A:[A_RIGHT,A_LEFT],B:list(range(10))} # For A, 0 is right and 1 is left

    def init(self):
        self.q1 = {MDP.A:np.zeros(len(MDP.ACTIONS[MDP.A])),MDP.B:np.zeros(len(MDP.ACTIONS[MDP.B]))}
        self.q2 = {MDP.A:np.zeros(len(MDP.ACTIONS[MDP.A])),MDP.B:np.zeros(len(MDP.ACTIONS[MDP.B]))}

    def epsilon_greedy(self, state, epsilon, q_values):
        # get an action index for a given state according to epsilon-greeedy strategy
        if np.random.binomial(1, epsilon) == 1:
            return np.random.choice(MDP.ACTIONS[state])
        else:
            max_action_value = np.max(q_values)
            action_ind = np.random.choice(
                [i for i in MDP.ACTIONS[state] if q_values[i] == max_action_value])
            return action_ind

    def step(self,state,action):
        if state==MDP.A:
            if action==MDP.A_RIGHT:#right
                return 0,MDP.END
            return 0, MDP.B
        if state==MDP.B:
            return np.random.randn()-0.1, MDP.END

    def q_learning(self,alpha,epsilon,gamma,double=False):
        state = MDP.A
        a_step_left = 0
        while state!=MDP.END:
            if double:
                action = self.epsilon_greedy(state, epsilon, self.q1[state] + self.q2[state])
            else:
                action = self.epsilon_greedy(state, epsilon, self.q1[state])
            reward, next_state = self.step(state,action)
            if state==MDP.A and action==MDP.A_LEFT:
                a_step_left+=1
            if double:
                if np.random.binomial(1,0.5)==1:
                    update_q = self.q1
                    max_q = self.q2
                else:
                    update_q = self.q2
                    max_q = self.q1
                if next_state!=MDP.END:
                    update_value = alpha*(reward+gamma*max_q[next_state][np.argmax(update_q[next_state])]-update_q[state][action])
                else:
                    update_value = alpha*(reward+gamma*0.0-update_q[state][action])
                update_q[state][action] += update_value
            else:
                if next_state!=MDP.END:
                    update_value = alpha*(reward+gamma*self.q1[next_state].max()-self.q1[state][action])
                else:
                    update_value = alpha*(reward+gamma*0.0-self.q1[state][action])
                self.q1[state][action]+=update_value
            state=next_state
        return a_step_left

    def figure_6_5(self):
        RUNS = 10000
        EPISODES = 300
        ALPHA = 0.1
        EPSILON = 0.1
        GAMMA = 1.0
        q_learning_doubles = [False,True]
        q_learning_names = ["Q-learning","Double Q-learning"]
        for double, name in zip(q_learning_doubles,q_learning_names):
            res = np.zeros((RUNS,EPISODES))
            for run in tqdm(range(RUNS)):
                self.init()
                for epi in range(EPISODES):
                    res[run][epi] = self.q_learning(ALPHA,EPSILON,GAMMA,double)
            plt.plot(np.arange(1,EPISODES+1),res.mean(axis=0),label=name)
        plt.plot(np.arange(1,EPISODES+1),np.ones(EPISODES)*0.05,linestyle=":", label="Optimal")
        plt.legend()
        plt.grid()
        plt.xlabel("Episodes")
        plt.ylabel("% left actions from A")
        plt.savefig("Figure_6_5.png")
        plt.close()
def figure_6_5():
    mdp = MDP()
    mdp.figure_6_5()

if __name__=="__main__":
    figure_6_5()
