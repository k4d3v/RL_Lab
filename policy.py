""" Represents a RBF policy"""

import numpy as np
from timeit import default_timer as timer

import torch


class Policy():
    def __init__(self, env, n_basis=50, dim_theta=305):
        """
        Nonlinear RBF network, used as a state-feedback controller
        :param env: Environment
        :param n_basis: Number of basis functions
        :param dim_theta: Dimension of each element of Theta
        """
        self.env = env
        self.s_dim = self.env.reset().shape[0]
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
        Samples a traj from performing actions based on the current policy
        :return: Sampled trajs
        """
        start = timer()

        # Reset the environment
        observation = self.env.reset()
        episode_reward = 0.0
        done = False
        traj = []

        while not done:
            # env.render()
            point = []

            action = self.get_policy(torch.Tensor(observation).view(self.s_dim, 1))

            point.append(observation)  # Save state to tuple
            point.append(action)  # Save action to tuple
            observation, reward, done, _ = self.env.step(action)  # Take action
            point.append(reward)  # Save reward to tuple

            episode_reward += reward
            traj.append(point)  # Add Tuple to traj

        end = timer()
        print("Done rollout, ", end - start)
        print("Episode reward: ", episode_reward)
        return traj

    def update(self, dJ):
        pass

    def check_convergence(self, old_Theta):
        return False
