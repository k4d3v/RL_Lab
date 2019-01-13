import numpy as np
from timeit import default_timer as timer
import torch


class Policy():
    """
    Represents a RBF policy
    """
    def __init__(self, env, n_basis=50, dim_theta=305):
        """
        Nonlinear RBF network, used as a state-feedback controller
        :param env: Environment
        :param n_basis: Number of basis functions
        :param dim_theta: Dimension of each element of Theta
        """
        self.env = env
        self.s_dim = self.env.observation_space.shape[0]
        self.n_basis = n_basis
        self.dim_theta = dim_theta # TODO: Why R^305!? (See paper)
        # Init. random control param.s
        W = np.random.normal(size=(self.n_basis))
        Lamb = np.zeros((self.s_dim, self.s_dim))
        np.fill_diagonal(Lamb, 1)
        Mu = np.random.normal(size=(self.n_basis, self.s_dim))
        self.Theta = {"W": W, "Lamb": Lamb, "Mu": Mu}

    def get_policy(self, x):
        """
        Returns a single control based on observation x
        :param x: Observation
        :return: Control
        """
        sum = 0
        for i in range(self.n_basis):
            sum += self.Theta["W"][i]*self.calc_feature(x, i)
        return np.array([sum])

    def calc_feature(self, x, i):
        """
        Calculates a basis function feature
        :param x: Observation
        :param i: Number of current basis function
        :return: phi_i(x)
        """
        phi_x = np.exp(-0.5*
                       np.dot(np.dot(np.transpose(x-self.Theta["Mu"][i]).T, np.linalg.inv(self.Theta["Lamb"])),
                              (x-self.Theta["Mu"][i])))
        return phi_x

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
            #self.env.render()
            point = []

            action = self.get_policy(np.asarray(observation))

            point.append(observation)  # Save state to tuple
            point.append(action)  # Save action to tuple
            observation, reward, done, _ = self.env.step(action)  # Take action
            point.append(reward)  # Save reward to tuple

            episode_reward += reward
            traj.append(point)  # Add Tuple to traj

        print("Done rollout, ", timer() - start)
        print("Episode reward: ", episode_reward)
        return traj

    def update(self, dJ):
        pass

    def check_convergence(self, old_Theta):
        return False
