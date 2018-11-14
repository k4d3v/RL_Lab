"""An implementation of a Polynomial Value Function with Ridge Regression"""
import numpy as np
import torch

# TODO: Remove HardCoded Stuff, wont work on other Envs
class ThreeLayerNet(torch.nn.Module):
    def __init__(self, D_in, H1, H2, D_out):
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        h_relu = torch.nn.functional.relu(self.linear1(x))
        h_relu2 = torch.nn.functional.relu(self.linear2(h_relu))
        y_pred = self.linear3(h_relu2)

        return y_pred


class ValueFunction:
    def __init__(self, discount):
        self.discount = discount
        self.model = ThreeLayerNet(3, 50, 25, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)


    def fit(self, trajs):
        """
        Estimates the value function using conjugate gradients
        :param traj: A set of sampled trajectories
        :return: The value function estimate for the given trajectories
        """

        tmp = self.empirical_reward(trajs[0])
        states = tmp[0]
        returns = tmp[1]

        x = torch.squeeze(torch.stack(states))
        y = torch.stack(returns)

        loss=0.0
        for t in range(1000):
            # Forward pass
            y_pred = self.model(x)

            # Compute loss
            loss = self.criterion(y_pred, y)
            #print(t, loss.item())

            # Zero gradients, perform backward pass, update weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print("Value function loss: ", loss.item())

    def empirical_reward(self, traj):
        """
        :param traj: A sampled trajectory (state, action, reward)
        :return: (state, empirical_return)
        """
        states = []
        rewards = []
        for i in range(len(traj)):
            reward=0.0
            for j in range(i, len(traj)):
                reward = reward + (self.discount**(j-i)) * traj[j][2]

            states.append(traj[i][0].view(1, 3))
            rewards.append(reward)

        return [states, rewards]







