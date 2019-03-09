import numpy as np
import torch
from timeit import default_timer as timer
from matplotlib import pyplot as plt


class Net(torch.nn.Module):
    """
    Represents a neural network with one hidden layer
    """
    def __init__(self, D_in, H1, D_out):
        """
        Initializes a NN with one hidden layer
        :param D_in: Input layer
        :param H1: Hidden layer 1
        :param D_out: Output layer
        """
        super(Net, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H1)
        self.linear2 = torch.nn.Linear(H1, D_out)

    def forward(self, x):
        """
        Predicts an output for a given input
        :param x: Input
        :return: Output
        """
        h_relu = torch.nn.functional.relu(self.linear1(x))
        y_pred = self.linear2(h_relu)

        return y_pred


class ValueFunction:
    """
    Represents an approximated value function
    """
    def __init__(self, s_dim, discount=0.99):
        """
        Initialies the value fun
        :param discount: Discount factor
        """
        self.discount = discount
        self.model = Net(s_dim, 20, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, trajs, init=False):
        """
        Fits the NN onto training data
        :param trajs: Trajs (state, action, reward) used as supervised learning training data
        """
        start = timer()

        # Fit for more epochs if net was newly initialized, else for less, as new vals are similar to init.
        epochs = 400 if init else 100
        batch_size = 64

        # Compute empirical reward based on trajs
        states, returns = self.empirical_reward(trajs)
        #states, returns = self.empirical_reward(trajs[:-1])
        #states_vali, returns_vali = self.empirical_reward([trajs[-1]])

        x = torch.Tensor(states)
        y = torch.Tensor(returns).view(len(returns), 1)
        #x_vali = torch.Tensor(states_vali)
        #y_vali = torch.Tensor(returns_vali).view(len(returns_vali), 1)

        #loss_list, val_list = [], []
        rng = range(epochs)
        for _ in rng:
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

            #loss_list.append(self.criterion(self.model(x), y).item())
            #val_list.append(self.criterion(self.model(x_vali), y_vali).item())

        # Plot n_eps-loss curves
        #self.plot_loss(rng, loss_list, val_list)

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

    def plot_loss(self, rng, loss_list, val_list):
        """
        Plots the training and validation loss
        :param rng: Range of epochs
        :param loss_list: Training loss
        :param val_list: Validation loss
        """
        plt.plot(rng, loss_list, label="Training Loss")
        plt.plot(rng, val_list, label="Validation Loss")
        plt.xlabel("n-eps")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
