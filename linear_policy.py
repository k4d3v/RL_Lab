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

    def update_params(self, step):
        # TODO: Remove HardCoded Stuff, wont work on other Envs

        new_param = self.W.data.numpy()+step[0:5].ravel()
        self.W = Variable(torch.from_numpy(new_param).float(), requires_grad=True)
        print("New param W: ", self.W)

        new_param_2 = self.b.data.numpy()+step[5].ravel()
        self.b = Variable(torch.from_numpy(new_param_2).float(), requires_grad=True)
        print("New param b: ", self.b)

        new_param_3 = self.std.data.numpy() + step[6].ravel()
        self.std = Variable(torch.from_numpy(new_param_3).float(), requires_grad=True)
        print("New param std: ", self.std)











