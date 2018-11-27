import torch
from torch.distributions import Normal


# TODO: Adjust params according to challenge
class RandomExplorationPolicy:
    """ Represents a random policy with mu and sigma as params"""

    def __init__(self, mu=0, Sigma=1):
        self.mu = mu
        self.Sigma = Sigma

    def get_dist(self, state):
        """
        Returns a normal distribution based on the state and current params
        :param state: The current state of the system
        :return: A torch normal distribution
        """
        mean = torch.mm(self.mu, state)
        dist = Normal(mean, self.Sigma)
        return dist

    def get_action(self, state):
        """
        Samples an action for the given state based on the policy
        :param state: Given state
        :return: Sampled action
        """
        dist = self.get_dist(state)
        return dist.sample().numpy()

    def update_params(self, mu_new, Sigma_new):
        """
        Updates the policy parameters
        :param step: Theta_k
        :return:
        """
        self.mu = mu_new
        self.Sigma = Sigma_new

    def get_params(self):
        """
        Returns the current policy params
        :return: A list with the policy params
        """
        return [self.mu, self.Sigma]