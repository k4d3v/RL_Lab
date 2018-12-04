import numpy as np
import torch
from scipy import spatial
import math
from timeit import default_timer as timer


class DynProg:
    """
    Represents the Dynamic Programming algorithm with its two cases Value and Policy Iteration
    """

    def __init__(self, model):
        """
        :param model: The dynamics and reward model
        """
        # Doc: States between (-pi,-8) and (pi,8) and action between -2 and 2
        self.model = model

    def train_val_iter(self, discount=0.9):
        """
        Value Iteration algo
        :param discount: Hyperparameter for weighting future rewards
        :return: Converged V function and states together with optimal policy
        """
        start = timer()
        # Init.
        oldvalues = np.zeros((self.model.n_states,))
        iter = 0
        while True:
            # print("Iteration ", iter)
            newvalues = np.zeros((self.model.n_states,))
            pred_acts = []

            # Iterate over states
            for s in range(self.model.n_states):
                Q_all = []
                # Iterate over actions
                for a in range(self.model.n_actions):
                    # Predict next state and reward for given action
                    nxt_state, idx = self.model.dynamics_matrix[s][a]
                    rew = self.model.reward_matrix[s][a]

                    # Compute Q and append
                    Q_all.append(rew + discount * oldvalues[idx])

                # Update V fun and policy
                Q_all = np.array(Q_all)
                newvalues[s] = np.max(Q_all)
                pred_acts.append(self.model.actions[np.argmax(Q_all)])

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
        start = timer()
        # Init
        Vk = np.zeros((self.model.n_states,))
        # Init. policy uniformly with actions
        policy = np.random.choice(range(self.model.n_actions), size=self.model.n_states)
        policy_old = np.copy(policy)

        round_num = 0
        # Repeat for policy convergence
        while True:
            # print("Round Number:", round_num)
            # Repeat for V convergence
            iter = 0
            while True:
                # print("Iteration ", iter)
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
                        Q_all[s][a] = reward + discount * Vk[idx]

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
