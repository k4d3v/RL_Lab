""" Main file for testing the implementation for the challenge."""

import gym
import quanser_robots
import dill
import pickle
import torch
import numpy as np
from build_model import ModelBuilder

from dyn_prog import DynProg
from fun_approximator import FitNN

# Solve for Pendulum
env = gym.make("Pendulum-v2")
#env = gym.make("Qube-v0")

# Dimension of states
s_dim = env.reset().shape[0]
reward = FitNN(s_dim+1, 1, env, False)
dynamics = FitNN(s_dim+1, s_dim, env, True)

# Sample training data
print("Rollout policy...")
points = reward.rollout(9000) # See plots in figures dir for optimal number of samples
max_rew = np.max([point[3] for point in points])

# Learn dynamics and rewards
reward.learn(points[:3000], 256, 64)
dynamics.learn(points, 256, 64)

# Save for later use
# TODO: Error when trying to save NN for Qube. Why?
pickle.dump(reward, open("nets/rew_Pendulum-v2.fitnn", 'wb'))
pickle.dump(dynamics, open("nets/dyn_Pendulum-v2.fitnn", 'wb'))

env_name = "Pendulum-v2"
#reward = pickle.load(open("nets/rew_" + env_name + ".fitnn", 'rb'))
#dynamics = pickle.load(open("nets/dyn_" + env_name + ".fitnn", "rb"))

sd = pickle.load(open("discretizations/sd_"+env_name + ".arr", 'rb'))
ad = pickle.load(open("discretizations/ad_"+env_name + ".arr", "rb"))

#sd = [100, 400]
#ad = [10, 20]

for s in sd:
    for a in ad:
        print("Building model")
        mb = ModelBuilder(env_name, (s, a))
        mb.build_model(reward, dynamics)
        mb.save_model()

print("Done")

# TODO: Compare results(plots and total reward) for different discretizations
# TODO: Plot best results of Value function and Policy for Pendulum - v2

# TODO: Swing up on Qube-v0
