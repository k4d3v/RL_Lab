import numpy as np
import torch
from torch.autograd import Variable
from torch.distributions import MultivariateNormal

class LinearPolicy:
    def __init__(self, action_dim, state_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.W = Variable(torch.rand(action_dim, state_dim), requires_grad=True)
        self.b = Variable(torch.rand(action_dim, 1), requires_grad=True)
        self.std = Variable(torch.eye(action_dim), requires_grad=True)

