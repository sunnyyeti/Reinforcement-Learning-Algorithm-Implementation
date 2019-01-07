import numpy as np
np.random.seed(1)
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn
HIT = 0
STICK = 1
class Card:
    CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    @classmethod
    def get_card(cls):
        return np.random.choice(cls.CARDS)

dealer_map_table = np.zeros(22,dtype=np.int32)
for i in range(17, 22):
    dealer_map_table[i] = STICK
for i in range(17):
    dealer_map_table[i] = HIT
def Dealer_Policy(dealer_state):
    return dealer_map_table[dealer_state]

player_policy_map_table = np.zeros(22, dtype=np.int32)  ##sum(12-21), dealer_show(1-10), usable_ace
player_policy_map_table[20] = player_policy_map_table[21] = STICK
def Player_Policy(player_state):
    return player_policy_map_table[player_state[0]]

class Participant:
    def __init__(self):
        self.policy = None
        self.sum = 0
        self.usable_ace = False
        self.burst = False
        self.iniaction = None
        self.dealer_show = None

    def set_policy(self, policy):
        self.policy = policy

    def hit(self):
        card = Card.get_card()
        # if card == 1:
        #     if self.sum + 11 > Participant.MAX:
        #         self.sum += 1
        #         if self.sum > Participant.MAX:
        #             if not self.usable_ace:
        #                 self.burst = True
        #             else:
        #                 self.sum-=10
        #                 self.usable_ace=False
        #     else:
        #         self.sum += 11
        #         self.usable_ace = True
        # else:
        #     if self.sum + card > Participant.MAX:
        #         if not self.usable_ace:
        #             self.burst = True
        #         else:
        #             self.sum += card - 10
        #             self.usable_ace = False
        #     else:
        #         self.sum += card

        ace_count = int(self.usable_ace)
        if card == 1:
            ace_count += 1
        self.sum += 11 if card==1 else card
        # If the player has a usable ace, use it as 1 to avoid busting and continue.
        while self.sum > 21 and ace_count:
            self.sum -= 10
            ace_count -= 1
        # player busts
        if self.sum > 21:
            self.burst=True

        self.usable_ace = (ace_count == 1)




class Player(Participant):
    def __init__(self):
        super(Player, self).__init__()

    def init(self, inistate=None, iniaction=None):
        if inistate is None:
            # generate a random initial state
            # initialize cards of player
            while self.sum < 12:
                # if sum of player is less than 12, always hit
                card = Card.get_card()
                self.sum += 11 if card==1 else card
                self.usable_ace = (card == 1)

            # Always use an ace as 11, unless there are two.
            # If the player's sum is larger than 21, he must hold two aces.
            if self.sum > 21:
                assert self.sum == 22
                # use one Ace as 1 rather than 11
                self.sum -= 10
        else:
            self.sum, self.dealer_show, self.usable_ace = inistate
        self.iniaction = iniaction
        self.burst = False

    def state(self):
        return self.sum, self.dealer_show, int(self.usable_ace)

    def run(self):
        trajectory = []
        act = self.policy(self.state()) if self.iniaction is None else self.iniaction
        trajectory.append((self.sum, self.dealer_show, int(self.usable_ace), act))
        while act == HIT:
            self.hit()
            if self.burst:
                break
            else:
                act = self.policy(self.state())
                trajectory.append((self.sum, self.dealer_show, int(self.usable_ace), act))
        return trajectory


class Dealer(Participant):
    def __init__(self):
        super(Dealer, self).__init__()
        #self.show = None
    def init(self, first_card=None):
        def card_value(card):
            return 11 if card==1 else card
        if first_card is None:
            dealer_card1 = Card.get_card()
        else:
            dealer_card1 = first_card
        dealer_card2 = Card.get_card()
        self.sum = card_value(dealer_card1) + card_value(dealer_card2)
        self.usable_ace = 1 in (dealer_card1, dealer_card2)
        # if the dealer's sum is larger than 21, he must hold two aces.
        if self.sum > 21:
            assert self.sum == 22
            # use one Ace as 1 rather than 11
            self.sum -= 10
        assert self.sum <= 21
        self.burst = False
        # self.show = first_card
        # if first_card == 1:
        #     self.sum = 11
        #     self.usable_ace = True
        # else:
        #     self.sum = first_card
        #     self.usable_ace = False
        # self.burst = False
        # self.iniaction = None


    def state(self):
        return self.sum

    def run(self):
        act = self.policy(self.state())
        while act == HIT:
            self.hit()
            if self.burst:
                break
            else:
                act = self.policy(self.state())



class Blackjack:
    def __init__(self):
        self.player = Player()
        self.dealer = Dealer()

    def state_value_estimation(self, episodes):
        class PlayerPolicy:
            def __init__(self, map_table):
                self.map_table = map_table

            def __call__(self, state):
                return self.map_table[state[0]]

        player_policy_map_table = np.zeros(22,dtype=np.int32)
        player_policy_map_table[20] = player_policy_map_table[21] = STICK
        self.player.set_policy(PlayerPolicy(player_policy_map_table))
        self.dealer.set_policy(Dealer_Policy)
        cur_sums = list(range(12, 22))
        show_cards = list(range(1, 11))
        usable_aces = [True, False]
        state_values = {True: np.zeros((10, 10), dtype=np.float64), False: np.zeros((10, 10), dtype=np.float64)}
        state_cnts = {True: np.zeros((10, 10), dtype=np.float64) + 1e-12, False: np.zeros((10, 10), dtype=np.float64)+1e-12}
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
        np.random.seed(1)
        self.dealer.set_policy(Dealer_Policy)
        self.player.set_policy(Player_Policy)
        action_values = np.zeros((10, 10, 2, 2), dtype=np.float32)  ##sum(12-21), dealer_show(1-10), usable_ace, action
        action_counts = np.zeros((10, 10, 2, 2), dtype=np.float32) + 1e-12
        cur_sums = list(range(12, 22))
        show_cards = list(range(1, 11))
        usable_aces = [False, True]
        actions = [0,1]
        def behavior_policy(state):
            player_sum, dealer_card, usable_ace  = state
            usable_ace = int(usable_ace)
            player_sum -= 12
            dealer_card -= 1
            # get argmax of the average returns(s, a)
            values_ = action_values[player_sum, dealer_card, usable_ace, :] / \
                      action_counts[player_sum, dealer_card, usable_ace, :]
            return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

        for _ in tqdm(range(episodes)):
            use_ace = np.random.choice(usable_aces)
            play_sum = np.random.choice(cur_sums)
            dealer_show = np.random.choice(show_cards)
            iniaction = np.random.choice(actions)
            inistate = (play_sum,dealer_show,use_ace)
            self.player.init(inistate,iniaction)
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
            for player_sum, dealer_card, player_useace, act in player_trajectory:
                action_values[player_sum-12,dealer_card-1,player_useace,act]+=reward
                action_counts[player_sum-12,dealer_card-1,player_useace,act]+=1.0
            #action_ave_reward = action_values/action_counts
            #new_policy = action_ave_reward.argmax(axis=-1)
            # new_policy = np.zeros_like(player_policy.map_table)
            # for i in range(10):
            #     for j in range(10):
            #         for k in range(2):
            #             values = action_ave_reward[i,j,k,:]
            #             action = np.random.choice([ a for a,v in enumerate(values) if v==np.max(values)])
            #             new_policy[i,j,k]=action
            self.player.set_policy(behavior_policy)
        #return action_values
        action_ave_reward = action_values / action_counts
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
        plt.subplot(2,2,i)
        policy = np.flipud(policy)
        fig = seaborn.heatmap(policy,cbar=True,cmap="YlGnBu",xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_xlabel(xlabel,fontsize=40)
        fig.set_ylabel(ylabel,fontsize=40)
        fig.set_title(title,fontsize=40)
    def draw_state_value(i,statevalue,xlabel,ylabel,title):
        ax = plt.subplot(2, 2, i, projection="3d")
        x = np.arange(1, 11)
        y = np.arange(12, 22)
        x, y = np.meshgrid(x, y)
        ax.plot_surface(x, y, statevalue, cmap=cm.coolwarm)
        ax.set_xlabel(xlabel,fontsize=40)
        ax.set_ylabel(ylabel,fontsize=40)
        ax.set_title(title,fontsize=40)

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
    plt.savefig("Figure_5_2_.png")
    plt.close()
if __name__ == "__main__":
    figure_5_1()
    figure_5_2()
