""" Main file for testing the implementation for the challenge."""

import gym
import quanser_robots
import dill
import pickle
import torch
import numpy as np

from dyn_prog import DynProg
from policy import RandomExplorationPolicy
from fun_approximator import FitNN

# Solve for Pendulum
#env = gym.make("Pendulum-v2")
env = gym.make("Qube-v0")
policy = RandomExplorationPolicy()

# Dimension of states
s_dim = env.reset().shape[0]
reward = FitNN(s_dim+1, 1, env, False)
dynamics = FitNN(s_dim+1, s_dim, env, True)

# Sample training data
print("Rollout policy...")
points = reward.rollout(15000) # See plots in figures dir for optimal number of samples

# Learn dynamics and rewards
reward.learn(points[:2000], 256, 64)
dynamics.learn(points, 256, 64)

# Save for later use
# TODO: Error when trying to save NN for Qube. Why?
pickle.dump(reward, open("nets/rew_Qube-v0.fitnn", 'wb'))
pickle.dump(dynamics, open("nets/dyn_Qube-v0.fitnn", 'wb'))

#agent = DynProg(policy, env, reward, dynamics)
#Vk, pol = agent.train_val_iter()
#Vk, pol = agent.train_pol_iter()

# TODO: Compare results(plots and total reward) for different discretizations
# TODO: Plot best results of Value function and Policy for Pendulum - v2

# TODO: Swing up on Qube-v0
