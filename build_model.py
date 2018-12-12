import numpy as np
import pickle
import gym
import quanser_robots
import matplotlib.pyplot as plt
import torch
from timeit import default_timer as timer


class ModelBuilder:
    """
    Represents a descrete model of the world (dynamics and reward)
    """

    def __init__(self, env_name):
        """
        :param env_name: Name of the learning environment
        :param reward: A NN trained on the reward function
        :param dynamics: A NN trained on the dynamics function
        :param n_sa: Tuple containing the number of states and actions
        """
        self.env_name = env_name

    def build_model(self, reward, dynamics, points, n_sa):
        """
        Stores each possible reward and next state based on the learnt models
        :param reward: A NN trained on rewards
        :param dynamics: A NN trained on dynamics
        """
        self.n_states = n_sa[0]
        self.n_actions = n_sa[1]
        # State space discretization
        # Doc: States between (-pi,-8) and (pi,8) and action between -2 and 2
        # TODO: Going beyond +-pi is outside of range of learnt reward and dynamics funs!  Maybe choose other vals
        # self.states = np.mgrid[-np.pi:np.pi:arg, -8.0:8.0:arg].reshape(2, -1).T
        self.states = np.array([p[0] for p in points[:n_sa[0]]])
        self.actions = np.linspace(-2, 2, n_sa[1])

        start = timer()
        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.dynamics_matrix = []

        # Predict each possible state and reward
        for s, state in zip(range(self.n_states), self.states):
            dynamics_row = []
            for a, action in zip(range(self.n_actions), self.actions):
                # Predict, transform and save reward
                rew = reward.predict(torch.Tensor([state[0], state[1], action]))
                self.reward_matrix[s][a] = np.abs(rew ** -1)

                # Predict, discretize and save state
                nxt_state = dynamics.predict(torch.Tensor([state[0], state[1], action]))
                idx = self.find_nearest(self.states, nxt_state)
                dynamics_row.append((self.states[idx], idx))

            self.dynamics_matrix.append(np.array(dynamics_row))
        self.dynamics_matrix = np.array(self.dynamics_matrix)
        print("Model ready", timer() - start)

    def find_nearest(self, array, value):
        """
        Finds the nearest discretized state
        :param array: Contains all discretized states
        :param value: A sampled state
        :return: Nearest state to value
        """
        return ((np.array(array) - value) ** 2).sum(1).argmin()

    def save_model(self):
        """
        Save model reward and dynamics matrix in files
        """
        pickle.dump(self,
                    open(
                        "models/" +
                        self.env_name + "_" + str(self.n_states) + "_" + str(self.n_actions) + ".mod", 'wb'))


def load_model(env_name, n_sa):
    """
    Load reward and dynamics matrix from files
    """
    return pickle.load(open(
        "models/" + env_name + "_" + str(n_sa[0]) + "_" + str(n_sa[1]) + ".mod", 'rb'))
