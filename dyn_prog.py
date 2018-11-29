import numpy as np
import torch


class DynProg:
    """
    Represents the Dynamic Programming algorithm with its two cases Value and Policy Iteration
    """

    def __init__(self, policy, env, reward, dynamics, n_samples=4):
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
        self.n_samples = n_samples
        # State space discretization
        # Doc: States between (-pi,-8) and (pi,8) and action between -2 and 2
        arg = np.sqrt(n_samples) * 1j
        self.states = np.mgrid[-np.pi:np.pi:arg, -8.0:8.0:arg].reshape(2, -1).T
        self.actions = np.linspace(-2, 2, n_samples)

    def train_val_iter(self, discount=0.1):
        """
        Value Iteration algo
        :param n_samples:
        :param discount:
        :return: Vk is the converged V function and policy is the optimal policy based on Vk
        """
        # Init.
        newvalues, oldvalues = np.zeros((self.n_samples, )), np.zeros((self.n_samples, ))
        cumul_reward = []

        iter = 0
        while True:
            print("Iteration ", iter)
            # Iterate over states
            for state, i in zip(self.states, range(self.states.shape[0])):
                Q_all = []
                # Iterate over actions
                for action in self.actions:
                    # Predict next state and reward for given action
                    nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                    reward = self.reward.predict(torch.Tensor([state[0], state[1], action]))

                    # Find nearest discrete state for predicted next state
                    idx = self.find_nearest(self.states, nxt_state)

                    # Compute Q and append
                    Q_all.append(reward + discount * oldvalues[idx])

                Q_all = np.array(Q_all)
                newvalues[i] = np.max(Q_all)

            cumul_reward.append(np.sum(newvalues))

            # Convergence check
            if (oldvalues == newvalues).all():
                break

            oldvalues = newvalues
            iter += 1

        cumul_reward = np.sum(np.array(cumul_reward))
        return newvalues, cumul_reward

    def train_pol_iter(self, n_samples=10000, discount=0.1):
        """
        Policy Iteration algo
        :param n_samples:
        :param discount:
        :return: Vk is the converged V function from the last iteration and policy is the converged policy
        """

        # TODO: Policy Iteration (30)
        # Init
        Vk, Vk_new = np.zeros((self.n_samples,)), np.zeros((self.n_samples,))
        policy = np.random.uniform(0, 1, len(self.states.shape[0]))
        is_stable = False
        round_num = 0
        # Repeat for policy convergence
        while not is_stable:
            is_stable = True
            round_num += 1
            print("Round Number:", round_num)
            round_num += 1
            # Repeat for V convergence
            iter = 0
            while True:
                print("Iteration ", iter)
                # Iterate over states
                for state, i in zip(self.states, range(self.states.shape[0])):
                    Q_all = []
                    # Iterate over actions
                    for action in self.actions:
                        # Predict next state and reward for given action
                        nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action]))
                        reward = self.reward.predict(torch.Tensor([state[0], state[1], action]))

                        # Find nearest discrete state for predicted next state
                        idx = self.find_nearest(self.states, nxt_state)

                        # Compute Q and append
                        Q_all.append(reward + discount * Vk[idx])

                    # Compute V-Function
                    Q_all = np.array(Q_all)
                    Vk_new[i] = np.sum(policy * Q_all)

                # Convergence check
                if (Vk == Vk_new).all():
                    break

                Vk = Vk_new
                iter += 1

            for state_num in range(self.states.shape[0]):
                action_by_policy = np.argmax(policy[state_num])
                best_action, best_action_value = self.next_best_action(state_num, Vk, discount)
                policy[state_num] = np.eye(self.n_samples)[best_action]
                if action_by_policy != best_action:
                    is_stable = False

        policy = [np.argmax(policy[state]) for state in range(self.states.shape[0])]
        return policy, Vk


    def max_action(self, V, R, discount):
        """Computes the V function. That is the maximum of the V parameter (Current possible rewards)
        """
        return max(V)

    def find_policy(self, V):
        """Finds an optimal policy for a state for 15 time steps, given the V function
        """
        return np.argmax(V)

    def calc_reward(self, currVt, s, a):
        """Calculates a Q value for a given state-action pair
        """
        # i = s[0] + a[0]
        # j = s[1] + a[1]
        # Return reward
        # return currVt[i][j]
        return 0


    def find_nearest(self, array, value):
        return ((array - value) ** 2).sum(1).argmin()

    def next_best_action(self, state, V, discount):
        action_values = np.zeros(self.n_samples)
        for action_num in range(self.n_samples):
            # Predict next state and reward for given action
            nxt_state = self.dynamics.predict(torch.Tensor([state[0], state[1], action_num]))
            reward = self.reward.predict(torch.Tensor([state[0], state[1], action_num]))

            # Find nearest discrete state for predicted next state
            idx = self.find_nearest(self.states, nxt_state)
            action_values[action_num] += (reward + discount * V[idx])
        return np.argmax(action_values), np.max(action_values)
