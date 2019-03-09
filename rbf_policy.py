import numpy as np
import torch
from torch.autograd import grad
from torch.autograd import Variable
from torch.distributions import Normal


class RBFPolicy:
    """ Represents an RBF policy
    like described in 'Towards Generalization and Simplicity in Continuous Control', p.4"""

    def __init__(self, env):
        torch.manual_seed(1)
        self.s_dim = env.observation_space.shape[0]
        self.min_a, self.max_a = env.action_space.low, env.action_space.high
        # Init W and bias to 0 and standard deviation such that 3*sigma=max_action
        self.W = [Variable(torch.rand(1), requires_grad=True) for d in range(self.s_dim)]
        self.b = Variable(torch.rand(1), requires_grad=True)
        self.std = Variable(torch.rand(1), requires_grad=True)
        self.v = 0.4
        self.P = [np.random.normal(size=(self.s_dim)) for j in range(self.s_dim)]
        self.phi = [np.random.uniform(-np.pi, np.pi) for j in range(self.s_dim)]

    def get_dist(self, state):
        """
        Returns a normal distribution based on the state and current params
        :param state: The current state of the system
        :return: A torch normal distribution
        """
        y = self.get_features(state)
        W = torch.stack(self.W).view(1, self.s_dim)
        mean = torch.mm(W, y) + self.b
        dist = Normal(mean, self.std)
        return dist

    def get_features(self, state):
        """
        Computes the features based on a given state
        :param state: Current state
        :return: Random Fourier features
        """
        y = [np.sin((np.inner(state.reshape(-1,), self.P[i])/self.v)+self.phi[i]) for i in range(self.s_dim)]
        return torch.Tensor(np.array(y).reshape(-1,1))

    def get_log_prob(self, state, action):
        """
        Calculates the log of the policy for an action conditioned on a state
        :param state: Current state
        :param action: Current action
        :return: log pi
        """
        dist = self.get_dist(state)
        return dist.log_prob(action)

    def get_gradient(self, state, action):
        """
        Calculates the gradient of the log of the conditioned policy
        :param state: Current state
        :param action: Current action
        :return: The gradient of log pi
        """
        dist = self.get_dist(state)
        grads = grad(dist.log_prob(action), (self.W + [self.b] + [self.std]))
        return np.array([x.numpy() for x in grads]).ravel()

    def get_gradient_analy(self, state, action):
        """
        Computes the log pi gradient analytically
        :param state: Current state
        :param action: Current action
        :return: Analytical log pi gradient
        """
        W = torch.stack(self.W).view(1, self.s_dim)
        mean = torch.mm(W, state) + self.b

        grad_W = [((1 / (self.std ** 2)) * (action - mean) * s).detach().numpy()[0][0] for s in state]
        grad_b = ((1 / (self.std ** 2)) * (action - mean)).detach().numpy()[0][0]
        grad_std = ((-1 / (2 * self.std ** 2)) * (1 - (1 / (self.std ** 2)) * (action - mean) ** 2) * 2 * self.std).detach().numpy()[0][0]
        return np.array(grad_W + [grad_b] + [grad_std])

    def get_action(self, state):
        """
        Samples an action for the given state based on the policy
        :param state: Given state
        :return: Sampled action
        """
        dist = self.get_dist(state)
        return dist.sample().numpy()[0]

    def update_params(self, step):
        """
        Updates the policy parameters
        :param step: Theta_k
        :return:
        """
        for i in range(len(step[:-2])):
            self.W[i].data += step[i]

        self.b.data += step[-2]
        self.std.data += step[-1]

    def get_params(self):
        """
        Returns the current policy params
        :return: A list with the policy params
        """
        return self.W + [self.b] + [self.std]

