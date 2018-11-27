""" Main file for testing the implementation for the challenge."""
import numbers

import gym
import quanser_robots

from dynprog import ValIter
from policy import RandomExplorationPolicy
from value_func import FitNN

# Solve for Pendulum
#env = gym.make("Pendulum-v0")
env = gym.make("Pendulum-v2")
policy = RandomExplorationPolicy()

# Dimension of states
s_dim = env.reset().shape[0]
reward = FitNN(s_dim+1, 1)
dynamics = FitNN(s_dim+1, s_dim)

agent = ValIter(policy, env, reward, dynamics)
agent.train()

