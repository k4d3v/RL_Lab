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


    def train_val_iter(self, discount=0.9):
        """
        Value Iteration algo
        :param n_states:
        :param discount:
        :return: Vk is the converged V function
        """
        # Init.
        oldvalues = np.zeros((self.n_states,))

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
                reward = np.abs(reward ** -1)
                idx = self.find_nearest(self.states, nxt_state)
                Q.append(reward + discount * oldvalues[idx])

            pred_act.append(self.actions[np.argmax(np.array(Q))])

        return newvalues, [self.states, pred_act]

    def train_pol_iter(self, discount=0.9):
        """
        Policy Iteration algo
        :param n_states:
        :param discount:
        :return: Vk is the converged V function from the last iteration and policy is the converged policy
        """

        # TODO: Policy Iteration (30)
        # Init
        Vk = np.zeros((self.n_states,))
        # Init. policy uniformly with actions
        policy = np.random.choice(range(self.n_actions), size=self.n_states)
        policy_old = policy[:]

        round_num = 0
        # Repeat for policy convergence
        while True:
            print("Round Number:", round_num)
            # Repeat for V convergence
            iter = 0
            policy = np.random.choice(range(self.n_actions), size=self.n_states)
            while True:
                print("Iteration ", iter)
                Vk_new = np.zeros((self.n_states,))
                Q_all = np.zeros((self.n_states, self.n_actions))
                # Iterate over states
                for state, i in zip(self.states, range(self.n_states)):
                    # Iterate over actions
                    for action, j in zip(self.actions, range(self.n_actions)):
                        # Predict next state and reward for given action
                        nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                        reward = self.reward.predict(torch.Tensor([state[0], state[1], action]))
                        reward = np.abs(reward**-1)

                        # Find nearest discrete state for predicted next state
                        idx = self.find_nearest(self.states, nxt_state)

                        # Compute Q and append
                        Q_all[i][j] = reward + discount * Vk[idx]

                    # Compute V-Function
                    Vk_new[i] = Q_all[i][policy[i]]
                    #print("hi")

                # Convergence check for V fun
                if (np.abs(Vk - Vk_new) < 0.1).all():
                    break

                Vk = Vk_new[:]
                iter += 1

            # Update policy
            for curr_s in range(self.n_states):
                policy[curr_s] = np.argmax(Q_all[curr_s][:])

            # Convergence check for policy
            if (policy_old == policy).all():
                break

            policy_old = policy[:]
            round_num += 1

        return Vk_new, [self.states, [self.actions[act] for act in policy]]

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
