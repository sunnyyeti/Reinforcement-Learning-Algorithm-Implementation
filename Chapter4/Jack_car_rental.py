import numpy as np
import seaborn
import matplotlib.pyplot as plt

class CarCompany():
    MAX_CARS = 20
    ACTIONS = np.arange(-5,6)
    def __init__(self):
        self.state_values = np.zeros((CarCompany.MAX_CARS+1,CarCompany.MAX_CARS+1),dtype=np.float32)
