import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class Card:
    CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    @classmethod
    def get_card(cls):
        return np.random.choice(cls.CARDS)

class Policy():
    def __init__(self,map_table):
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

    def run(self):
        trajectory = []
        if self.burst:
            return
        act = self.policy(self.state) if self.iniaction is None else self.iniaction
        trajectory.append((self.sum, self.usable_ace, act))
        while act == Participant.HIT:
            self.hit()
            if self.burst:
                break
            else:
                act = self.policy(self.state)
                trajectory.append((self.sum, self.usable_ace, act))
        return trajectory


class Player(Participant):
    def __init__(self):
        super(Player,self).__init__()

    def init(self, inisum, usable_ace, iniaction):
        self.sum = inisum
        self.usable_ace = usable_ace
        self.iniaction = iniaction
        self.burst = False

    def set_dealer(self,dealer):
        self.dealer = dealer

    @property
    def state(self):
        return self.sum,  self.dealer.show, int(self.usable_ace),

class Dealer(Participant):
    def __init__(self):
        super(Dealer,self).__init__()

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

class Blackjack:
    def __init__(self):
        self.player = Player()
        self.dealer = Dealer()
        self.player.set_dealer(self.dealer)

    def state_value_estimation(self,episodes):
        class State_Policy(Policy):
            def __init__(self,map_table):
                super(State_Policy,self).__init__(map_table)
            def __call__(self, state):
                return self.map_table[state[0]]

        player_policy = np.ones(Participant.MAX + 1)
        player_policy[20] = player_policy[21] = Participant.STICK
        dealer_policy = np.ones(Participant.MAX + 1)
        for i in range(17, Participant.MAX + 1):
            dealer_policy[i] = Participant.STICK

        self.player.set_policy(State_Policy(player_policy))
        self.dealer.set_policy(State_Policy(dealer_policy))
        cur_sums = list(range(12,22))
        show_cards =  list(range(1,11))
        usable_aces =[True,False]
        state_values = {True:np.zeros((10,10),dtype=np.float64),False:np.zeros((10,10),dtype=np.float64)}
        state_cnts = {True:np.zeros((10,10),dtype=np.float64),False:np.zeros((10,10),dtype=np.float64)}
        for _ in tqdm(range(episodes)):
            play_sum = np.random.choice(cur_sums)
            dealer_show = np.random.choice(show_cards)
            use_ace = np.random.choice(usable_aces)
            self.player.init(play_sum,use_ace,None)
            self.dealer.init(dealer_show)
            player_trajectory = self.player.run()
            if self.player.burst:
                reward = -1
            else:
                self.dealer.run()
                if self.dealer.burst or self.player.sum>self.dealer.sum:
                    reward = 1
                elif self.player.sum<self.dealer.sum:
                    reward = -1
                else:
                    reward = 0
            for player_sum, player_useace, _ in player_trajectory:
                state_values[player_useace][play_sum-12][self.dealer.show-1]+=reward
                state_cnts[player_useace][play_sum-12][self.dealer.show-1]+=1.0
        state_values[True] = state_values[True]/state_cnts[True]
        state_values[False] = state_values[False]/state_cnts[False]
        return state_values

    def action_value_estimation_with_exploring_states(self):
        class Player_Policy(Policy):
            def __init__(self,map_table):
                super(Player_Policy,self).__init__(map_table)
            def __call__(self, state):
                return self.map_table[state]




def figure_5_1():
    def draw(i,value,xlabel,ylabel,title):
        ax = plt.subplot(2, 2, i, projection="3d")
        x = np.arange(1,11)
        y = np.arange(12,22)
        x,y = np.meshgrid(x,y)
        ax.plot_surface(x, y, value, cmap=cm.coolwarm)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    blackjack = Blackjack()
    plt.figure(figsize=(20,20))
    state_values = blackjack.state_value_estimation(10000)
    xlabel_ = "Dealer showing"
    ylabel_ = "Player sum"
    draw(1,state_values[True],xlabel_,ylabel_,"Usable ace; After 10000 episodes")
    draw(3,state_values[False],xlabel_,ylabel_,"No usable ace; After 10000 episodes")
    state_values = blackjack.state_value_estimation(500000)
    draw(2,state_values[True],xlabel_,ylabel_,"Usable ace; After 500000 episodes")
    draw(4,state_values[False],xlabel_,ylabel_,"No usable ace; After 500000 episodes")
    plt.savefig("Figure_5_1.png")
    plt.close()

if __name__ == "__main__":
    figure_5_1()
