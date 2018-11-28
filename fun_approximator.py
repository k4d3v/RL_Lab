import numpy as np
import torch
from timeit import default_timer as timer


# TODO: Remove HardCoded Stuff. Only usable with action_dim=1 and observation_dim=5
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


class FitNN:
    """
    Fit a NN to given points
    """
    def __init__(self, input, output):
        """
        Initialize the NN
        :param input: Input dimension
        :param output: Output dimension
        """
        self.model = ThreeLayerNet(input, 100, 100, output)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4)

    def fit_batch(self, x, y, epochs, batch_size):
        """
        Fits the NN onto training data points
        """
        start = timer()
        print("Total-Loss before Fitting: ", self.validate_model(x, y))

        for _ in range(epochs):
            perm = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = perm[i:i + batch_size]
                batch_x = x[indices]
                batch_y = y[indices]

                batch_pred = self.model(batch_x)
                loss = self.criterion(batch_pred, batch_y)

                loss.backward()
                self.optimizer.step()

        end = timer()
        self.total_loss = self.validate_model(x, y)
        print("Total-Loss after Fitting: ", self.total_loss)
        print("Done fitting! Time elapsed: ", end - start)

    def validate_model(self, x, y):
        return self.criterion(self.model(x), y).item()

    def rollout(self, num, env):
        """
        Rolls out the policy for num timesteps
        """
        points = []
        points_collected = 0
        while points_collected < num:

            # Reset the environment
            observation = env.reset()
            done = False

            while not done and points_collected < num:
                old_observation = observation
                action = env.action_space.sample()
                observation, reward, done, _ = env.step(action)  # Take action
                points.append([old_observation, observation, action, reward])
                points_collected += 1

        return points

    def learn(self, dyn, points):
        """
        Fit a model for predicting reward or dynamics
        :param dyn: If True, learn dynamics. Else learn reward
        :param points: Trajectory samples for learning
        """
        # Generate Points by rolling out the policy

        # Prepare the points for the NN
        x, y = [], []
        for point in points:
            old, new, act, rew = point
            x.append(np.append(old, act))
            y.append(new if dyn else rew)
        x = torch.Tensor(x)
        y = torch.Tensor(y) if dyn else torch.Tensor(y).view(len(y), 1)

        # Fit wanted function
        print("-----------------------------------------------------")
        word = "dynamics" if dyn else "reward"
        print("Fit " + word + " Function... This will take around x sec")
        self.fit_batch(x, y, 1000, 256)
