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
        arg = np.sqrt(n_samples)*1j
        self.states = np.mgrid[-np.pi:np.pi:arg, -8.0:8.0:arg].reshape(2, -1).T
        self.actions = np.linspace(-2, 2, n_samples)

    def train_val_iter(self, discount=0.1):
        """
        Value Iteration algo
        :param n_samples:
        :param discount:
        :return: Vk is the converged V function and policy is the optimal policy based on Vk
        """

        oldvalues = np.zeros((self.n_samples,))
        newvalues = np.zeros((self.n_samples,))
        cumul_reward = []

        for _ in range(10):
            print("Value iteration ", _)
            for i in range(self.n_samples):
                Q_all = []
                for action in self.actions:
                    # Predict next state and reward for given action
                    nxt_state = self.dynamics.predict(torch.Tensor([self.states[i][0], self.states[i][1], action]))
                    reward = self.reward.predict(torch.Tensor([self.states[i][0], self.states[i][1], action]))

                    # Find nearest discrete state for predicted next state
                    idx = self.find_nearest(self.states, nxt_state)

                    # Compute Q
                    Q = reward + discount * oldvalues[idx]
                    Q_all.append(Q)

                Q_all = np.array(Q_all)
                newvalues[i] = self.actions[Q_all.max()]

            cumul_reward.append(np.sum(newvalues))
            oldvalues = newvalues

        return 0, 0

    def train_pol_iter(self, n_samples=10000, discount=0.1):
        """
        Policy Iteration algo
        :param n_samples:
        :param discount:
        :return: Vk is the converged V function from the last iteration and policy is the converged policy
        """

        # TODO: Reward matrix
        R = np.zeros((1, 1))
        x, y = R.shape[0], R.shape[1]

        # TODO: Policy Iteration (30)

        # Init
        Vk = np.zeros((x, y))
        policy = np.random.uniform(0, 1)

        # Repeat for policy convergence
        while True:
            # Repeat for V convergence
            while True:
                Vk_new = np.zeros((x, y))
                currQ = []
                a_curr = []
                a_pre = []
                a_pre = a_curr
                policy_curr = policy
                for i in range(x):
                    for j in range(y):
                        # TODO: Compute Q function
                        # currQ = []
                        for a in self.actions:
                            currQ.append(R[i][j] + discount * self.calc_reward(Vk, [i, j], a))
                            a_curr.append(a)

                        # Compute V function
                        Vk_new = + (policy * currQ)

                # Check convergence
                if (Vk == Vk_new).all():
                    break
                Vk = Vk_new

                maxQindex = np.argmax(currQ)

                if a_pre == a_curr[maxQindex]:
                    policy = 1
                else:
                    policy = 0

            # TODO: Policy convergence check
            if (policy_curr == policy):
                break

        # TODO: Compute the cumulative reward over 100 episodes

        return Vk, policy

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
        #i = s[0] + a[0]
        #j = s[1] + a[1]
        # Return reward
        #return currVt[i][j]
        return 0

    def find_nearest(self, array, value):
        return ((array - value) ** 2).sum(1).argmin()
