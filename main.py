""" Main file for testing the implementation for the challenge."""
import random

import gym
import quanser_robots
import dill
import pickle
import torch
import numpy as np
from build_model import ModelBuilder

from dyn_prog import DynProg
from fun_approximator import FitNN

random.seed(42)

# Solve for Pendulum
env = gym.make("Pendulum-v2")
#env = gym.make("Qube-v0")

# Dimension of states
s_dim = env.reset().shape[0]
reward = FitNN(s_dim+1, 1, env, False)
dynamics1 = FitNN(s_dim+1, s_dim, env, True)
dynamics2 = FitNN(s_dim+1, s_dim, env, True, True)

# Sample training data
print("Rollout policy...")
points = reward.rollout(3000) # See plots in figures dir for optimal number of samples
#max_rew = np.max([point[3] for point in points])
# Learn dynamics and rewards
reward.learn(points, 256, 64)
#dynamics.learn(points, 256, 64)

points_left = dynamics1.rollout(18000)
points_right = dynamics2.rollout(18000)

dynamics1.learn(points_left, 1024, 64)
dynamics2.learn(points_right, 1024, 64)

o1, o2 = 0,0
for pl, pr in zip(points_left, points_right):
    predl = dynamics1.predict(torch.Tensor([pl[0][0], pl[0][1], pl[2][0]]))
    predr = dynamics2.predict(torch.Tensor([pr[0][0], pr[0][1], pr[2][0]]))
    reall = pl[1]
    realr = pr[1]
    if np.sum(np.abs(predl-reall))>1:
        o1+=1
    if np.sum(np.abs(predr-realr))>1:
        o2+=1
print(o1)
print(o2)

# Save for later use
# TODO: Error when trying to save NN for Qube. Why?
pickle.dump(reward, open("nets/rew_Pendulum-v2.fitnn", 'wb'))
pickle.dump(dynamics1, open("nets/dyn1_Pendulum-v2.fitnn", 'wb'))
pickle.dump(dynamics2, open("nets/dyn2_Pendulum-v2.fitnn", 'wb'))

env_name = "Pendulum-v2"
#reward = pickle.load(open("nets/rew_" + env_name + ".fitnn", 'rb'))
#dynamics1 = pickle.load(open("nets/dyn1_" + env_name + ".fitnn", "rb"))
#dynamics2 = pickle.load(open("nets/dyn2_" + env_name + ".fitnn", "rb"))

#a = np.abs(reward.predict(torch.Tensor([0, 0, 0]))**-1)

sd = pickle.load(open("discretizations/sd_"+env_name + ".arr", 'rb'))
ad = pickle.load(open("discretizations/ad_"+env_name + ".arr", "rb"))

points = reward.rollout(10000)
for s in sd:
    for a in ad:
        print("Building model")
        mb = ModelBuilder(env_name, reward, dynamics1, dynamics2)
        mb.build_model(points, (s, a))
        mb.save_model()

print("Done")

# TODO: Compare results(plots and total reward) for different discretizations
# TODO: Plot best results of Value function and Policy for Pendulum - v2

# TODO: Swing up on Qube-v0
