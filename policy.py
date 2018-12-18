""" Represents a policy"""
import numpy as np


class Policy():
    def __init__(self, n_basis=50, dim_theta=305):
        """
        Nonlinear RBF network, used as a state-feedback controller
        :param n_basis: Number of basis functions
        :param dim_theta: Dimension of each element of Theta
        """
        self.n_basis = n_basis
        self.dim_theta = dim_theta
        # Init. random control param.s
        W = np.random.normal(size=(self.n_basis, self.dim_theta))
        Lamb = np.random.normal(size=(self.dim_theta, self.dim_theta))
        Mu = np.random.normal(size=(self.n_basis, self.dim_theta))
        self.Theta = {"W": W, "Lamb": Lamb, "Mu": Mu}

    def get_policy(self, x):
        sum = 0
        for i in range(self.n_basis):
            sum += self.Theta["W"][i]*self.calc_feature(x, i)
        return sum

    def calc_feature(self, x, i):
        return np.exp(-0.5*(x-self.Theta["Mu"][i]).T*np.linalg.inv(self.Theta["Lamb"])*(x-self.Theta["Mu"][i]))

    def rollout(self):
        """
        Rolls out a policy
        :param policy: Current policy
        :return: Sampled data points
        """
        data = 0
        return data

    def update(self, dJ):
        pass

    def check_convergence(self, old_Theta):
        return False
