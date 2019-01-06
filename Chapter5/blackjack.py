import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn

class Card:
    CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    @classmethod
    def get_card(cls):
        return np.random.choice(cls.CARDS)


class Policy():
    def __init__(self, map_table):
        self.map_table = map_table


class Participant:
    HIT = 1
    STICK = 0
    MAX = 21

    def __init__(self):
        self.policy = None
        self.sum = 0
        self.usable_ace = False
        self.burst = False
        self.iniaction = None

    def set_policy(self, policy):
        self.policy = policy

    def hit(self):
        card = Card.get_card()
        if card == 1:
            if self.sum + 11 > Participant.MAX:
                self.sum += 1
                if self.sum > Participant.MAX:
                    self.burst = True
            else:
                self.sum += 11
                self.usable_ace = True
        else:
            if self.sum + card > Participant.MAX:
                if not self.usable_ace:
                    self.burst = True
                else:
                    self.sum += card - 10
                    self.usable_ace = False
            else:
                self.sum += card




class Player(Participant):
    def __init__(self):
        super(Player, self).__init__()

    def init(self, inisum, usable_ace, iniaction):
        self.sum = inisum
        self.usable_ace = usable_ace
        self.iniaction = iniaction
        self.burst = False

    def set_dealer(self, dealer):
        self.dealer = dealer

    @property
    def state(self):
        return self.sum, self.dealer.show, int(self.usable_ace)

    def run(self):
        trajectory = []
        if self.burst:
            return
        act = self.policy(self.state) if self.iniaction is None else self.iniaction
        trajectory.append((self.sum, self.dealer.show, int(self.usable_ace), act))
        while act == Participant.HIT:
            self.hit()
            if self.burst:
                break
            else:
                act = self.policy(self.state)
                trajectory.append((self.sum, self.dealer.show, int(self.usable_ace), act))
        return trajectory


class Dealer(Participant):
    def __init__(self):
        super(Dealer, self).__init__()
        self.show = None
    def init(self, first_card):
        self.show = first_card
        if first_card == 1:
            self.sum = 11
            self.usable_ace = True
        else:
            self.sum = first_card
            self.usable_ace = False
        self.burst = False
        self.iniaction = None

    @property
    def state(self):
        return self.sum, int(self.usable_ace)

    def run(self):
        if self.burst:
            return
        act = self.policy(self.state) if self.iniaction is None else self.iniaction
        while act == Participant.HIT:
            self.hit()
            if self.burst:
                break
            else:
                act = self.policy(self.state)



class Blackjack:
    def __init__(self):
        self.player = Player()
        self.dealer = Dealer()
        self.player.set_dealer(self.dealer)

    def state_value_estimation(self, episodes):
        class State_Policy(Policy):
            def __init__(self, map_table):
                super(State_Policy, self).__init__(map_table)

            def __call__(self, state):
                return self.map_table[state[0]]

        player_policy = np.ones(Participant.MAX + 1,dtype=np.int8)
        player_policy[20] = player_policy[21] = Participant.STICK
        dealer_policy = np.ones(Participant.MAX + 1,dtype=np.int8)
        for i in range(17, Participant.MAX + 1):
            dealer_policy[i] = Participant.STICK

        self.player.set_policy(State_Policy(player_policy))
        self.dealer.set_policy(State_Policy(dealer_policy))
        cur_sums = list(range(12, 22))
        show_cards = list(range(1, 11))
        usable_aces = [True, False]
        state_values = {True: np.zeros((10, 10), dtype=np.float64), False: np.zeros((10, 10), dtype=np.float64)}
        state_cnts = {True: np.zeros((10, 10), dtype=np.float64) + 1e-12, False: np.zeros((10, 10), dtype=np.float64)}
        for _ in tqdm(range(episodes)):
            play_sum = np.random.choice(cur_sums)
            dealer_show = np.random.choice(show_cards)
            use_ace = np.random.choice(usable_aces)
            self.player.init(play_sum, use_ace, None)
            self.dealer.init(dealer_show)
            player_trajectory = self.player.run()
            if self.player.burst:
                reward = -1
            else:
                self.dealer.run()
                if self.dealer.burst or self.player.sum > self.dealer.sum:
                    reward = 1
                elif self.player.sum < self.dealer.sum:
                    reward = -1
                else:
                    reward = 0
            for player_sum, dealer_show, player_useace, _ in player_trajectory:
                state_values[player_useace][play_sum - 12][dealer_show - 1] += reward
                state_cnts[player_useace][play_sum - 12][dealer_show - 1] += 1.0
        state_values[True] = state_values[True] / state_cnts[True]
        state_values[False] = state_values[False] / state_cnts[False]
        return state_values

    def action_value_estimation_with_exploring_states(self,episodes):
        class Player_Policy(Policy):
            def __init__(self, map_table):
                super(Player_Policy, self).__init__(map_table)

            def __call__(self, state):
                return self.map_table[state[0]-12, state[1]-1, state[2]]

            def update_policy(self, map_table):
                self.map_table = map_table

        class Dealer_Policy(Policy):
            def __init__(self, map_table):
                super(Dealer_Policy, self).__init__(map_table)

            def __call__(self, state):
                return self.map_table[state[0]]

        dealer_policy = np.ones(Participant.MAX + 1,dtype=np.int8)
        for i in range(17, Participant.MAX + 1):
            dealer_policy[i] = Participant.STICK
        dealer_policy = Dealer_Policy(dealer_policy)
        self.dealer.set_policy(dealer_policy)
        player_policy = np.ones((10,10,2),dtype=np.int8) ##sum(12-21), dealer_show(1-10), usable_ace
        player_policy[-2:,:,:] = Participant.STICK
        player_policy = Player_Policy(player_policy)
        self.player.set_policy(player_policy)
        action_values = np.zeros((10, 10, 2, 2), dtype=np.float64)  ##sum(12-21), dealer_show(1-10), usable_ace, action
        action_counts = np.zeros((10, 10, 2, 2), dtype=np.float64) + 1e-12
        cur_sums = list(range(12, 22))
        show_cards = list(range(1, 11))
        usable_aces = [True, False]
        actions = [Participant.HIT,Participant.STICK]
        for _ in tqdm(range(episodes)):
            play_sum = np.random.choice(cur_sums)
            use_ace = np.random.choice(usable_aces)
            iniaction = np.random.choice(actions)
            self.player.init(play_sum, use_ace, iniaction)
            dealer_show = np.random.choice(show_cards)
            self.dealer.init(dealer_show)
            player_trajectory = self.player.run()
            if self.player.burst:
                reward = -1
            else:
                self.dealer.run()
                if self.dealer.burst or self.player.sum > self.dealer.sum:
                    reward = 1
                elif self.player.sum < self.dealer.sum:
                    reward = -1
                else:
                    reward = 0
            for player_sum, dealer_show, player_useace, act in player_trajectory:
                action_values[play_sum-12,dealer_show-1,player_useace,act]+=reward
                action_counts[play_sum-12,dealer_show-1,player_useace,act]+=1.0
            action_ave_reward = action_values/action_counts
            new_policy = action_ave_reward.argmax(axis=-1)
            # new_policy = np.zeros_like(player_policy.map_table)
            # for i in range(10):
            #     for j in range(10):
            #         for k in range(2):
            #             values = action_ave_reward[i,j,k,:]
            #             action = np.random.choice([ a for a,v in enumerate(values) if v==np.max(values)])
            #             new_policy[i,j,k]=action
            player_policy.update_policy(new_policy)
        return action_ave_reward



def figure_5_1():
    # def draw(i, value, xlabel, ylabel, title):
    #     ax = plt.subplot(2, 2, i)
    #     policy = np.flipud(value)
    #     fig = seaborn.heatmap(policy,cbar=True,cmap="YlGnBu",xticklabels=range(1, 11),
    #                       yticklabels=list(reversed(range(12, 22))))
    #     fig.set_xlabel(xlabel)
    #     fig.set_ylabel(ylabel)
    #     fig.set_title(title)

    def draw(i, value, xlabel, ylabel, title):
        ax = plt.subplot(2, 2, i, projection="3d")
        x = np.arange(1, 11)
        y = np.arange(12, 22)
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, value, cmap=cm.coolwarm)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    blackjack = Blackjack()
    plt.figure(figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    state_values = blackjack.state_value_estimation(10000)
    xlabel_ = "Dealer showing"
    ylabel_ = "Player sum"
    draw(1, state_values[True], xlabel_, ylabel_, "Usable ace; After 10000 episodes")
    draw(3, state_values[False], xlabel_, ylabel_, "No usable ace; After 10000 episodes")
    state_values = blackjack.state_value_estimation(500000)
    draw(2, state_values[True], xlabel_, ylabel_, "Usable ace; After 500000 episodes")
    draw(4, state_values[False], xlabel_, ylabel_, "No usable ace; After 500000 episodes")
    plt.savefig("Figure_5_1.png")
    plt.close()

def figure_5_2():
    def draw_policy_heatmap(i,policy,xlabel,ylabel,title):
        ax=plt.subplot(2,2,i)
        policy = np.flipud(policy)
        fig = seaborn.heatmap(policy,cbar=True,cmap="YlGnBu",xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_xlabel(xlabel)
        fig.set_ylabel(ylabel)
        fig.set_title(title)
    def draw_state_value(i,statevalue,xlabel,ylabel,title):
        ax = plt.subplot(2, 2, i, projection="3d")
        x = np.arange(1, 11)
        y = np.arange(12, 22)
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, statevalue, cmap=cm.coolwarm)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    blackjack = Blackjack()
    plt.figure(figsize=(40,30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    action_ave_values = blackjack.action_value_estimation_with_exploring_states(1000000)##sum, show, usable_ace, act
    usable_ace_action_values = action_ave_values[:,:,1,:]
    no_usable_ace_action_values = action_ave_values[:,:,0,:]
    draw_policy_heatmap(1,np.argmax(usable_ace_action_values,axis=-1),"Dealer showing","Player sum","Usable ace; Policy")
    draw_policy_heatmap(3,np.argmax(no_usable_ace_action_values,axis=-1),"Dealer showing","Player sum","No usable ace; Policy")
    draw_state_value(2,usable_ace_action_values.max(axis=-1),"Dealer showing","Player sum","Usable ace; State value")
    draw_state_value(4,no_usable_ace_action_values.max(axis=-1),"Dealer showing","Player sum","No usable ace; State value")
    plt.savefig("Figure_5_2.png")
    plt.close()
if __name__ == "__main__":
    #figure_5_1()
    figure_5_2()
