import numpy as np
import torch


# TODO: Remove HardCoded Stuff. Only usable with action_dim=1 and observation_dim=5
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
    def __init__(self, discount=0.95):
        self.discount = discount
        self.model = ThreeLayerNet(5, 50, 25, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def fit(self, trajs):
        epoches = 3
        batch_size = 64

        states, returns = self.empirical_reward(trajs)

        x = torch.Tensor(states)
        y = torch.Tensor(returns).view(len(returns), 1)

        for _ in range(epoches):
            perm = torch.randperm(x.size()[0])
            for i in range(0, x.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = perm[i:i+batch_size]
                batch_x = x[indices]
                batch_y = y[indices]

                batch_pred = self.model(batch_x)
                loss = self.criterion(batch_pred, batch_y)

                loss.backward()
                self.optimizer.step()

        #pred = self.model(x)
        #print("Value-Function-Loss: ", self.criterion(pred, y).item())

    def empirical_reward(self, trajs):
        """
        :param traj: A sampled trajectory (state, action, reward)
        :return: (state, empirical_return)
        """
        states = []
        rewards = []

        for traj in trajs:
            for i in range(len(traj)):
                reward=0.0
                for j in range(i, len(traj)):
                    reward += (self.discount**(j-i)) * traj[j][2]

                states.append(np.array(traj[i][0]))
                rewards.append(reward)

        return [states, rewards]

    def predict(self, trajs):
        all_values=[]
        for traj in trajs:
            traj_values=[]
            for timestep in traj:
                state = torch.Tensor(timestep[0])
                traj_values.append(self.model(state).detach().numpy())
            all_values.append(traj_values)
        return all_values
