import numpy as np
import torch
from timeit import default_timer as timer
from matplotlib import pyplot as plt


class ThreeLayerNet(torch.nn.Module):
    """
    Represents a neural network with three layers
    """
    def __init__(self, D_in, H1, H2, D_out):
        """
        Initializes a NN with three layers
        :param D_in: Input layer
        :param H1: Hidden layer 1
        :param H2: Hidden layer 2
        :param D_out: Output layer
        """
        super(ThreeLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, H2)
        self.linear3 = torch.nn.Linear(H2, D_out)

    def forward(self, x):
        """
        Predicts an output for a given input
        :param x: Input
        :return: Output
        """
        h_relu = torch.nn.functional.relu(self.linear1(x))
        h_relu2 = torch.nn.functional.relu(self.linear2(h_relu))
        y_pred = self.linear3(h_relu2)

        return y_pred


class ValueFunction:
    """
    Represents an approximated value function
    """
    def __init__(self, s_dim, discount=0.95):
        """
        Initialies the value fun
        :param discount: Discount factor
        """
        self.discount = discount
        self.model = ThreeLayerNet(s_dim, 50, 25, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, trajs, init=False):
        """
        Fits the NN onto training data
        :param trajs: Trajs (state, action, reward) used as supervised learning training data
        """
        start = timer()

        # Fit for more epochs if net was newly initialized, else for less, as new vals are similar to init.
        epochs = 200 if init else 20
        batch_size = 64

        # Compute empirical reward based on trajs
        states, returns = self.empirical_reward(trajs)

        x = torch.Tensor(states)
        y = torch.Tensor(returns).view(len(returns), 1)

        for _ in range(epochs):
            perm = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = perm[i:i+batch_size]
                batch_x = x[indices]
                batch_y = y[indices]

                batch_pred = self.model(batch_x)
                # Compute loss based on true data and prediction
                loss = self.criterion(batch_pred, batch_y)

                loss.backward()
                self.optimizer.step()

        pred = self.model(x)
        print("Value-Function-Loss: ", self.criterion(pred, y).item())

        # Plot ground truth and prediction
        # Uncomment for debugging
        #self.plot(pred, y, len(trajs[0]))
        #self.plot(pred, y)

        print("Done fitting, ", timer() - start)

    def empirical_reward(self, trajs):
        """
        :param traj: A sampled trajectory (state, action, reward)
        :return: (state, empirical_return)
        """
        start = timer()

        states = []
        rewards = []
        for traj in trajs:
            for i in range(len(traj)):
                reward=0.0
                for j in range(i, len(traj)):
                    reward += (self.discount**(j-i)) * traj[j][2]

                states.append(np.array(traj[i][0]))
                rewards.append(reward)
        print("Done emp. reward, ", timer() - start)
        return [states, rewards]

    def predict(self, trajs):
        """
        Predicts some rewards based on the input trajectories
        :param trajs:
        :return: Rewards for each time step on every traj
        """
        all_values=[]
        for traj in trajs:
            traj_values=[]
            for timestep in traj:
                state = torch.Tensor(timestep[0])
                # Predict output for each timestep
                traj_values.append(self.model(state).detach().numpy())
            all_values.append(traj_values)
        return all_values

    def plot(self, pred, y, n=0):
        """
        Plots gorund truth of V vs prediction
        :param n: Number of points to plot
        :param pred: Predicted vals of regression model
        :param y: Ground truth
        """
        if n == 0:
            x = range(len(y))
            n = len(y)
        else:
            x = range(n)
        pred = pred.detach().numpy().reshape(-1, )
        y = y.detach().numpy().reshape(-1, )
        plt.plot(x, pred[:n], label="Prediction")
        plt.plot(x, y[:n], label="Ground Truth")
        plt.legend()
        plt.xlabel("State Number")
        plt.ylabel("Empirical Reward")
        plt.title("Empirical Reward: Ground Truth vs Prediction")
        plt.show()
