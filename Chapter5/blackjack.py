import numpy as np
import matplotlib.pyplot as plt

class Card:
    CARDS = [1,2,3,4,5,6,7,8,9,10,10,10,10]
    @classmethod
    def get_card(cls):
        return np.random.choice(cls.CARDS)

class Participant:
    HIT = 1
    STICK = 0
    MAX = 21
    def __init__(self):
        self.policy = None
        self.sum = 0
        self.usable_ace=False
        self.burst = False
        self.iniaction = None

    def set_policy(self,policy):
        self.policy = policy

    def hit(self):
        card = Card.get_card()
        if card==1:
            if self.sum+11>Participant.MAX:
                self.sum+=1
                if self.sum>Participant.MAX:
                    self.burst = True
            else:
                self.sum+=11
                self.usable_ace = True
        else:
            if self.sum+card>Participant.MAX:
                if not self.usable_ace:
                    self.burst=True
                else:
                    self.sum +=card-10
                    self.usable_ace=False
            else:
                self.sum+=card

    def run(self):
        trajectory = []
        if self.burst:
            return
        act = self.policy[self.sum] if self.iniaction is None else self.iniaction
        trajectory.append((self.sum,self.usable_ace,act))
        while act==Participant.HIT:
            self.hit()
            if self.burst:
                break
            else:
                act = self.policy[self.sum]
                trajectory.append((self.sum,self.usable_ace,act))
        return trajectory


class Player(Participant):
    def __init__(self):
        super().__init__()

    def _init(self,inisum,usable_ace,iniaction):
        self.sum = inisum
        self.usable_ace = usable_ace
        self.iniaction = iniaction
        self.burst = False


class Dealer(Participant):
    def __init__(self):
        super().__init__()

    def _init(self,first_card):
        self.show = first_card
        if first_card==1:
            self.sum = 11
            self.usable_ace = True
        else:
            self.sum = first_card
            self.usable_ace = False
        self.burst = False
        self.iniaction = None


class Blackjack:
    def __init__(self):
        self.player = Player()
        self.dealer = Dealer()

    def state_value_estimation(self):
        player_policy = np.ones(Participant.MAX+1)
        player_policy[20]=player_policy[21]=Participant.STICK
        dealer_policy = np.ones(Participant.MAX+1)
        for i in range(17,Participant.MAX+1):
            dealer_policy[i] = Participant.STICK


