
#CAN BE DELETED

import matplotlib.pyplot as plt

class Logger:
    def __init__(self):
        self.bla = 0
        self.iteration = []
        self.reward = []

    def add(self, rew):
        self.iteration.append(self.bla)
        self.bla = self.bla+1
        self.reward.append(rew)

    def print(self):
        print(self.iteration)
        print(self.reward)

    def plot(self):
        plt.plot(self.iteration, self.reward)
        plt.show()

