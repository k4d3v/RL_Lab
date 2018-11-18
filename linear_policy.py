import numpy as np
import torch
from torch.autograd import grad
from torch.autograd import Variable
from torch.distributions import Normal


# TODO: Remove HardCoded Stuff. Only usable with action_dim=1 and observation_dim=5
class SimpleLinearPolicy:
    def __init__(self):
        self.w1 = Variable(torch.rand(1), requires_grad=True)
        self.w2 = Variable(torch.rand(1), requires_grad=True)
        self.w3 = Variable(torch.rand(1), requires_grad=True)
        self.w4 = Variable(torch.rand(1), requires_grad=True)
        self.w5 = Variable(torch.rand(1), requires_grad=True)
        self.b = Variable(torch.rand(1), requires_grad=True)
        self.std = Variable(torch.rand(1), requires_grad=True)

    def get_dist(self, state):
        W = torch.stack([self.w1, self.w2, self.w3, self.w4, self.w5]).view(1, 5)
        mean = torch.mm(W, state) + self.b
        dist = Normal(mean, self.std)
        return dist

    def get_log_prob(self, state, action):
        dist = self.get_dist(state)
        return dist.log_prob(action)

    def get_gradient(self, state, action):
        dist = self.get_dist(state)
        grads = grad(dist.log_prob(action), [self.w1, self.w2, self.w3, self.w4, self.w5, self.b, self.std])
        return np.array([x.numpy() for x in grads]).ravel()

    def get_gradient_analy(self, state, action):
        W = torch.stack([self.w1, self.w2, self.w3, self.w4, self.w5]).view(1, 5)
        mean = torch.mm(W, state) + self.b

        grad_w1 = (1/(self.std**2)) * (action - mean) * state[0]
        grad_w2 = (1 / (self.std ** 2)) * (action - mean) * state[1]
        grad_w3 = (1 / (self.std ** 2)) * (action - mean) * state[2]
        grad_w4 = (1 / (self.std ** 2)) * (action - mean) * state[3]
        grad_w5 = (1 / (self.std ** 2)) * (action - mean) * state[4]
        grad_b = (1/(self.std**2)) * (action - mean)
        grad_std = (-1/(2*self.std**2)) * (1 - (1/(self.std**2)) * (action - mean)**2) * 2*self.std

        return np.array([grad_w1.detach().numpy()[0][0], grad_w2.detach().numpy()[0][0], grad_w3.detach().numpy()[0][0], grad_w4.detach().numpy()[0][0], grad_w5.detach().numpy()[0][0], grad_b.detach().numpy()[0][0], grad_std.detach().numpy()[0][0]])

    def get_action(self, state):
        dist = self.get_dist(state)
        return dist.sample().numpy()

    def update_params(self, step):
        self.w1.data += step[0]
        self.w2.data += step[1]
        self.w3.data += step[2]
        self.w4.data += step[3]
        self.w5.data += step[4]
        self.b.data += step[5]
        self.std.data += step[6]

    def get_params(self):
        return [self.w1, self.w2, self.w3, self.w4, self.w5, self.b, self.std]

