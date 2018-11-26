""" Main file for testing the implementation for the challenge."""

import gym

from dynprog import ValIter
from policy import RandomExplorationPolicy
from value_func import ValueFunction

# Solve for Pendulum
env = gym.make("Pendulum-v0")
policy = RandomExplorationPolicy()
val = ValueFunction()

agent = ValIter(policy, env, val)
agent.train()
#TODO

# Solve for Qube
env = gym.make("Qube-v0")
#TODO