from value_func import FitNN
import torch
import numpy as np


class ValIter:
    def __init__(self, policy, env, reward, dynamics):
        self.policy = policy
        self.env = env
        self.reward = reward
        self.dynamics = dynamics

    def train(self):
        # Generate Points by rolling out the policy
        print("1. Rollout policy...")
        points = self.rollout(10000)

        # Prepare the points for the NN
        x, y, y1 = [], [], []
        for point in points:
            old, new, act, rew = point
            x.append(np.append(old, act))
            y.append(rew)
            y1.append(new)
        x = torch.Tensor(x)
        y = torch.Tensor(y).view(len(y), 1)
        y1 = torch.Tensor(y1)

        # Fit reward dunction
        print("-----------------------------------------------------")
        print("2. Fit reward Function... This will take around 30 sec")
        self.reward.fit_batch(x, y, 1000, 256)

        # Fit dynamics function
        print("-------------------------------------------------------")
        print("3. Fit dynamics Function... This will take around 30 sec")
        self.dynamics.fit_batch(x, y1, 1000, 256)

        # Perform DP with Value Iteration / Policy Iteration
        # TODO

        # Compute cumulative reward over 100 episodes
        # TODO

        return 0


    def rollout(self, num):
        """
        Rolls out the policy for num timesteps
        """
        points = []
        points_collected = 0
        while points_collected < num:

            # Reset the environment
            observation = self.env.reset()
            done = False

            while not done and points_collected < num:
                old_observation = observation
                action = self.env.action_space.sample()
                observation, reward, done, _ = self.env.step(action)  # Take action
                points.append([old_observation, observation, action, reward])
                points_collected += 1

        return points




