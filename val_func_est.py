"""An implementation of a Polynomial Value Function with Ridge Regression"""

class ValueFunction:
    def __init__(self, n, discount):
        self.n = n
        self.discount = discount


    def fit(self, traj):
        """
        Estimates the value function using conjugate gradients
        :param traj: A set of sampled trajectories
        :return: The value function estimate for the given trajectories
        """
        return 0

    def empirical_reward(self, traj):
        """
        :param traj: A sampled trajectory (state, action, reward)
        :return: (state, empircal_return)
        """

        print("Test")

        for i in range(len(traj)):
            print(traj[i][0].numpy())







