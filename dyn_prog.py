import numpy as np
import torch
from scipy import spatial
import math
from timeit import default_timer as timer


class DynProg:
    """
    Represents the Dynamic Programming algorithm with its two cases Value and Policy Iteration
    """

    def __init__(self, env, reward, dynamics, n_sa=(2500, 8)):
        """
        :param env: The learning environment
        :param reward: A NN trained on the reward function
        :param dynamics: A NN trained on the dynamics function
        """
        self.env = env
        self.reward = reward
        self.dynamics = dynamics
        self.n_states = n_sa[0]
        self.n_actions = n_sa[1]
        # State space discretization
        # Doc: States between (-pi,-8) and (pi,8) and action between -2 and 2
        arg = np.sqrt(self.n_states) * 1j
        # TODO: Going beyond +-pi is outside of range of learnt reward and dynamics funs!  Maybe choose other vals
        self.states = np.mgrid[-np.pi:np.pi:arg, -8.0:8.0:arg].reshape(2, -1).T
        self.actions = np.linspace(-2, 2, n_sa[1])
        self.build_model()

    def train_val_iter(self, discount=0.9):
        """
        Value Iteration algo
        :param discount: Hyperparameter for weighting future rewards
        :return: Converged V function and states together with optimal policy
        """
        start = timer()
        # Init.
        oldvalues = np.zeros((self.n_states,))
        iter = 0
        while True:
            # print("Iteration ", iter)
            newvalues = np.zeros((self.n_states,))
            pred_acts = []

            # Iterate over states
            for s in range(self.n_states):
                Q_all = []
                # Iterate over actions
                for a in range(self.n_actions):
                    # Predict next state and reward for given action
                    nxt_state, idx = self.dynamics_matrix[s][a]
                    rew = self.reward_matrix[s][a]

                    # Compute Q and append
                    Q_all.append(rew + discount * oldvalues[idx])

                # Update V fun and policy
                Q_all = np.array(Q_all)
                newvalues[s] = np.max(Q_all)
                pred_acts.append(self.actions[np.argmax(Q_all)])

            # Convergence check
            if (np.abs(oldvalues - newvalues) < 0.1).all():
                print("val_iter done", timer() - start)
                break

            oldvalues = newvalues[:]
            iter += 1

        return newvalues, [self.states, pred_acts]

    def train_pol_iter(self, discount=0.9):
        """
        Policy Iteration algo
        :param discount: Hyperparameter for weighting future rewards
        :return: Converged V function and states together with optimal policy
        """
        start = timer()
        # Init
        Vk = np.zeros((self.n_states,))
        # Init. policy uniformly with actions
        policy = np.random.choice(range(self.n_actions), size=self.n_states)
        policy_old = np.copy(policy)

        round_num = 0
        # Repeat for policy convergence
        while True:
            # print("Round Number:", round_num)
            # Repeat for V convergence
            iter = 0
            while True:
                # print("Iteration ", iter)
                Vk_new = np.zeros((self.n_states,))
                Q_all = np.zeros((self.n_states, self.n_actions))
                # Iterate over states
                for s in range(self.n_states):
                    # Iterate over actions
                    for a in range(self.n_actions):
                        # Predict next state and reward for given action
                        nxt_state, idx = self.dynamics_matrix[s][a]
                        reward = self.reward_matrix[s][a]

                        # Compute Q and append
                        Q_all[s][a] = reward + discount * Vk[idx]

                    # Compute V-Function
                    Vk_new[s] = Q_all[s][policy[s]]

                # Convergence check for V fun
                if (np.abs(Vk - Vk_new) < 0.1).all():
                    break

                Vk = Vk_new[:]
                iter += 1

            # Update policy
            for curr_s in range(self.n_states):
                policy[curr_s] = np.argmax(Q_all[curr_s][:])

            # Convergence check for policy
            if (np.abs(policy_old - policy) < 0.1 * self.n_actions).all():
                print("pol_iter done", timer() - start)
                break

            policy_old = np.copy(policy)
            round_num += 1

        return Vk_new, [self.states, [self.actions[act] for act in policy]]

    def find_nearest(self, array, value):
        """
        Finds the nearest discretized state
        :param array: Contains all discretized states
        :param value: A sampled state
        :return: Nearest state to value
        """
        return ((array - value) ** 2).sum(1).argmin()

    def build_model(self):
        """
        Stores each possible reward and next state based on the learnt models
        """
        self.reward_matrix = np.zeros((self.n_states, self.n_actions))
        self.dynamics_matrix = []

        # Predict each possible state and reward
        for s, state in zip(range(self.n_states), self.states):
            dynamics_row = []
            for a, action in zip(range(self.n_actions), self.actions):
                # Predict, transform and save reward
                rew = self.reward.predict(torch.Tensor([state[0], state[1], action]))
                self.reward_matrix[s][a] = np.abs(rew ** -1)

                # Predict, discretize and save state
                nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                idx = self.find_nearest(self.states, nxt_state)
                dynamics_row.append((self.states[idx], idx))

            self.dynamics_matrix.append(np.array(dynamics_row))
        self.dynamics_matrix = np.array(self.dynamics_matrix)
