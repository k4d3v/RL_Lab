""" Represents a dynamic model based on a GP"""
import numpy as np


class DynModel:
    def __init__(self, s_dim):
        self.s_dim = s_dim

    def train(self, data):
        # Number of data points
        self.big_t = len(data[0])
        self.mu = np.zeros((self.big_t, self.s_dim))
        Sigma = []
        for t in range(self.big_t):
            Sigma_t = np.zeros((self.s_dim, self.s_dim))
            np.fill_diagonal(Sigma_t, 1)
            Sigma.append(Sigma_t)
        self.Sigma = Sigma
        print("hi")