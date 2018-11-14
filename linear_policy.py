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
        self.params = [self.W, self.b, self.std]

    def get_action(self, state):
        mean = torch.mm(self.W, state) + self.b
        mean = mean.view(1, self.action_dim)
        gauss = MultivariateNormal(mean, self.std)
        return gauss.sample()

    def get_log_prob(self, state, action):
        mean = torch.mm(self.W, state) + self.b
        mean = mean.view(1, self.action_dim)
        gauss = MultivariateNormal(mean, self.std)
        return gauss.log_prob(action)

    def get_gradient(self, state, action):
        prob = self.get_log_prob(state, action)

        if(type(self.W.grad).__name__ != 'NoneType'):
            self.W.grad.data.zero_()
            self.b.grad.data.zero_()
            self.std.grad.data.zero_()

        prob.backward()

        # Flatten gradients
        # TODO: Look for a better way to flatten
        grads = []
        for x in self.W.grad.numpy():
            for y in x:
                grads.append(y)

        for x in self.b.grad.numpy():
            for y in x:
                grads.append(y)

        for x in self.std.grad.numpy():
            for y in x:
                grads.append(y)

        return np.array(grads)

    def get_params(self):
        return self.params

"""
state_dimension = 2
action_dimension = 2

policy = LinearPolicy(action_dimension, state_dimension)

out = policy.get_action(torch.randint(-10, 10, (state_dimension, 1)))
print(out)

grad = policy.get_gradient(torch.randint(-10, 10, (state_dimension, 1)), torch.Tensor([0.0, 0.0]))
print(grad)

"""
"""
out = policy.get_log_prob(torch.randint(-10, 10, (state_dimension, 1)), torch.Tensor([0.0, 0.0]))
print(out.detach().numpy())


out.backward()
p = policy.get_params()
print(p[0].grad)
print(p[1].grad)
print(p[2].grad)
p[0].grad.data.zero_()
p[1].grad.data.zero_()"""








