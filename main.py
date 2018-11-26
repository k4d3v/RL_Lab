""" Main file for testing the implementation for the challenge."""

import gym

from dynprog import ValIter
from policy import RandomExplorationPolicy
from value_func import FitNN

# Solve for Pendulum
env = gym.make("Pendulum-v0")
policy = RandomExplorationPolicy()
reward = FitNN(4, 1)
dynamics = FitNN(4, 3)

agent = ValIter(policy, env, reward, dynamics)
agent.train()

