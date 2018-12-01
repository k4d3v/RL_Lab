import numpy as np
import torch
from scipy import spatial
import math

class DynProg:
    """
    Represents the Dynamic Programming algorithm with its two cases Value and Policy Iteration
    """

    def __init__(self, policy, env, reward, dynamics, n_sa=(16, 4)):
        """
        :param policy: A Gaussian exploration policy
        :param env: The learning environment
        :param reward: A NN trained on the reward function
        :param dynamics: A NN trained on the dynamics function
        """
        self.policy = policy
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


    def train_val_iter(self, discount=0.5):
        """
        Value Iteration algo
        :param n_states:
        :param discount:
        :return: Vk is the converged V function
        """
        # Init.
        oldvalues = np.zeros((self.n_states,))
        cumul_reward = []

        iter = 0
        while True:
            print("Iteration ", iter)
            newvalues = np.zeros((self.n_states,))

            # Iterate over states
            for state, i in zip(self.states, range(self.n_states)):
                Q_all = []
                # Iterate over actions
                for action in self.actions:
                    # Predict next state and reward for given action
                    nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                    rew = self.reward.predict(torch.Tensor([state[0], state[1], action]))
                    rew = np.abs(rew ** -1)

                    # Find nearest discrete state for predicted next state
                    idx = self.find_nearest(self.states, nxt_state)

                    # Compute Q and append
                    Q_all.append(rew + discount * oldvalues[idx])

                newvalues[i] = np.max(np.array(Q_all))

            # Convergence check
            if (np.abs(oldvalues - newvalues) < 0.1).all():
                break

            oldvalues = newvalues[:]
            iter += 1


        # Generate policy based on converged Value Function
        pred_act = []
        for state in self.states:
            Q = []
            for action in self.actions:
                nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                reward = self.reward.predict(torch.Tensor([state[0], state[1], action]))
                idx = self.find_nearest(self.states, nxt_state)
                Q.append(reward + discount * oldvalues[idx])

            pred_act.append(self.actions[np.argmax(np.array(Q))])

        return newvalues, [self.states, pred_act]

    def train_pol_iter(self, n_states=10000, discount=0.1):
        """
        Policy Iteration algo
        :param n_states:
        :param discount:
        :return: Vk is the converged V function from the last iteration and policy is the converged policy
        """

        # TODO: Policy Iteration (30)
        # Init
        Vk, Vk_new = np.zeros((self.n_states,)), np.zeros((self.n_states,))
        policy = np.tile(np.eye(self.n_actions)[1], (self.n_states, 1))
        is_stable = False
        round_num = 0

        # Repeat for policy convergence
        while not is_stable:
            is_stable = True
            print("Round Number:", round_num)
            round_num += 1
            # Repeat for V convergence
            iter = 0
            cumul_reward = []
            while True:
                print("Iteration ", iter)
                # Iterate over states
                for state, i in zip(self.states, range(self.n_states)):
                    Q_all = []
                    # Iterate over actions
                    for action in self.actions:
                        # Predict next state and reward for given action
                        nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                        reward = self.reward.predict(torch.Tensor([state[0], state[1], action]))

                        # Find nearest discrete state for predicted next state
                        idx_old = self.find_nearest(self.states, state)
                        idx = self.find_nearest(self.states, nxt_state)

                        # Compute Q and append
                        Q_all.append(reward + discount * Vk[idx])

                    # Compute V-Function
                    Q_all = np.array(Q_all)
                    Vk_new[i] = np.sum(policy[i] * Q_all)

                print(np.sum(Vk_new))
                cumul_reward.append(np.sum(Vk_new) / self.n_states)
                print(cumul_reward[iter])

                # Convergence check
                if (np.abs(Vk - Vk_new) < 0.1).all():
                    break

                Vk = Vk_new[:]
                iter += 1
                cumul_reward = np.sum(np.array(cumul_reward)) / iter if iter > 0 else np.sum(np.array(cumul_reward))

            # for state_num in range(self.n_states):
            for state, state_num in zip(self.states, range(self.n_states)):
                action_by_policy = np.argmax(policy[state_num])
                best_action, best_action_value = self.next_best_action(state, Vk, discount)
                policy[state_num] = np.eye(self.n_actions)[best_action]
                if action_by_policy != best_action:
                    is_stable = False

        policy = [np.argmax(policy[state_num]) for state_num in range(self.n_states)]
        return policy, Vk

    def find_nearest(self, array, value):
        return ((array - value) ** 2).sum(1).argmin()

    def next_best_action(self, state, V, discount):
        action_values = np.zeros(self.n_actions)
        for action_num in range(self.n_actions):
            # Predict next state and reward for given action
            nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action_num]))
            reward = self.reward.predict(torch.Tensor([state[0], state[1], action_num]))

            # Find nearest discrete state for predicted next state
            idx = self.find_nearest(self.states, nxt_state)
            action_values[action_num] += (reward + discount * V[idx])
        return np.argmax(action_values), np.max(action_values)
