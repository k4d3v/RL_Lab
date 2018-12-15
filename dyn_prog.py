from heapq import nsmallest

import numpy as np
import torch
from scipy import spatial
import math
from timeit import default_timer as timer
import scipy.stats


class DynProg:
    """
    Represents the Dynamic Programming algorithm with its two cases Value and Policy Iteration
    """

    def __init__(self, model, n_prob=4):
        """
        :param model: The dynamics and reward model
        """
        self.model = model
        normal = scipy.stats.norm(0, 1)
        self.n_prob = n_prob
        # Gauss between mu and 3*sigma
        step = 3/self.n_prob
        self.gauss = [normal.pdf(sn*step) for sn in range(self.n_prob)]

    def train_val_iter(self, discount=0.9):
        """
        Value Iteration algo
        :param discount: Hyperparameter for weighting future rewards
        :return: Converged V function and states together with optimal policy
        """
        print("Running value iteration...")
        start = timer()
        # Init.
        oldvalues = np.zeros((self.model.n_states,))
        iter = 0
        while True:
            print("Iteration ", iter)
            newvalues = np.zeros((self.model.n_states,))
            pred_acts = np.zeros(self.model.n_states)

            # Iterate over states
            for s in range(self.model.n_states):
                Q_all = np.zeros(self.model.n_actions)
                # Iterate over actions
                for a in range(self.model.n_actions):
                    # Predict next state and reward for given action
                    nxt_state, idx = self.model.dynamics_matrix[s][a]
                    rew = self.model.reward_matrix[s][a]

                    """
                    rew_tru = np.abs((-(self.model.states[s][0]**2
                                        + 0.1*self.model.states[s][1]**2
                                        + 0.001*self.model.actions[a]))**-1)

                    best_state_index = ((self.model.states - [0,0]) ** 2).sum(1).argmin()
                    r_max_pred = \
                        self.model.reward_matrix[best_state_index][1]
                    r_max = np.abs((-(0**2
                                        + 0.1*0**2
                                        + 0.001*0)+0.000000001)**-1)
                    """
                    # Compute Q and append
                    Q_all[a] = rew + discount * self.gauss_sum(nxt_state, oldvalues, idx)

                # Update V fun and policy
                newvalues[s] = np.max(Q_all)
                pred_acts[s] = self.model.actions[np.argmax(Q_all)]

            # Convergence check
            if (np.abs(oldvalues - newvalues) < 0.1).all():
                print("val_iter done", timer() - start)
                break

            oldvalues = newvalues[:]
            iter += 1

        return newvalues, [self.model.states, pred_acts]

    def train_pol_iter(self, discount=0.9):
        """
        Policy Iteration algo
        :param discount: Hyperparameter for weighting future rewards
        :return: Converged V function and states together with optimal policy
        """
        print("Running policy iteration...")
        start = timer()
        # Init
        Vk = np.zeros((self.model.n_states,))
        # Init. policy uniformly with actions
        policy = np.random.choice(range(self.model.n_actions), size=self.model.n_states)
        policy_old = np.copy(policy)

        round_num = 0
        # Repeat for policy convergence
        while True:
            print("Round Number:", round_num)
            # Repeat for V convergence
            iter = 0
            while True:
                print("Iteration ", iter)
                Vk_new = np.zeros((self.model.n_states,))
                Q_all = np.zeros((self.model.n_states, self.model.n_actions))
                # Iterate over states
                for s in range(self.model.n_states):
                    # Iterate over actions
                    for a in range(self.model.n_actions):
                        # Predict next state and reward for given action
                        nxt_state, idx = self.model.dynamics_matrix[s][a]
                        reward = self.model.reward_matrix[s][a]

                        # Compute Q and append
                        Q_all[s][a] = reward + discount * self.gauss_sum(nxt_state, Vk, idx)

                    # Compute V-Function
                    Vk_new[s] = Q_all[s][policy[s]]

                # Convergence check for V fun
                if (np.abs(Vk - Vk_new) < 0.1).all():
                    break

                Vk = Vk_new[:]
                iter += 1

            # Update policy
            for curr_s in range(self.model.n_states):
                policy[curr_s] = np.argmax(Q_all[curr_s][:])

            # Convergence check for policy
            if (np.abs(policy_old - policy) < 0.1 * self.model.n_actions).all():
                print("pol_iter done", timer() - start)
                break

            policy_old = np.copy(policy)
            round_num += 1

        return Vk_new, [self.model.states, [self.model.actions[act] for act in policy]]

    def gauss_sum(self, nxt_state, oldvalues, idx):
        """
        Computes a weighted sum over values
        :param nxt_state: Next state after an action
        :param oldvalues: Vk_old
        :return: Weighted sum over values for 5 neighbouring states
        """
        total = self.gauss[0]*oldvalues[idx]
        dists = np.delete(((self.model.states - nxt_state) ** 2).sum(1), idx, 0)
        for x in range(1, self.n_prob):
            # Find state with closest dist
            idx = np.argmin(dists)

            # Add weighted value to sum
            total += self.gauss[x] * oldvalues[idx]

            # Cut dist
            dists = np.delete(dists, idx, 0)
        return total
