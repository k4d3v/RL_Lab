import numpy as np


class DynProg:
    """
    Represents the Dynamic Programming algorithm with its two cases Value and Policy Iteration
    """

    def __init__(self, policy, env, reward, dynamics):
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

    def train_val_iter(self, n_samples=10000, discount=0.1):
        """
        Value Iteration algo
        :param n_samples:
        :param discount:
        :return: Vk is the converged V function and policy is the optimal policy based on Vk
        """

        # Compute cumulative reward over 100 episodes
        # TODO

        # TODO: State space discretization (?)

        # TODO: Reward matrix
        R = np.zeros((1, 1))
        x, y = R.shape[0], R.shape[1]

        # TODO: Value Iteration (33)

        # Init. episode
        init_state = self.env.reset()

        actions = [self.policy.get_action(init_state) for i in range(n_samples)]

        policy = np.full((x, y), 0)

        # Init
        Vk = np.zeros((x, y))

        # Repeat
        while True:
            Vk_new = np.zeros((x, y))
            for i in range(x):
                for j in range(y):

                    # Compute Q function
                    currQ = []
                    for a in actions:
                        currQ.append(R[i][j] + discount * self.calc_reward(Vk, [i, j], a))

                    # Compute V function
                    Vk_new[i][j] = self.max_action(currQ, R, 0)

                    # Update policy
                    policy[i][j] = self.find_policy(currQ)

            # Check convergence
            if (Vk == Vk_new).all():
                break
            Vk = Vk_new

        # TODO: Compute the cumulative reward over 100 episodes

        return Vk, policy

    def train_pol_iter(self, n_samples=10000, discount=0.1):
        """
        Policy Iteration algo
        :param n_samples:
        :param discount:
        :return: Vk is the converged V function from the last iteration and policy is the converged policy
        """

        # Compute cumulative reward over 100 episodes
        # TODO

        # TODO: State space discretization (?)

        # TODO: Reward matrix
        R = np.zeros((1, 1))
        x, y = R.shape[0], R.shape[1]

        # TODO: Policy Iteration (30)

        # Init. episode
        init_state = self.env.reset()

        actions = [self.policy.get_action(init_state) for i in range(n_samples)]

        policy = np.full((x, y), 0)

        # Init
        Vk = np.zeros((x, y))

        # Repeat for policy convergence
        while True:
            # Repeat for V convergence
            while True:
                Vk_new = np.zeros((x, y))
                for i in range(x):
                    for j in range(y):

                        # TODO: Compute Q function
                        currQ = []
                        for a in actions:
                            currQ.append(R[i][j] + discount * self.calc_reward(Vk, [i, j], a))

                        # Compute V function
                        Vk_new[i][j] = self.max_action(currQ, R, 0)

                        # Update policy
                        policy[i][j] = self.find_policy(currQ)

                # Check convergence
                if (Vk == Vk_new).all():
                    break
                Vk = Vk_new

            # TODO: Policy convergence check
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
        i = s[0] + a[0]
        j = s[1] + a[1]
        # Return reward
        return currVt[i][j]
