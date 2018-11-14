""" Main file for testing the implementation of the NPG algorithm"""

import npg
import gym
import linear_policy
import val_func_est


env = gym.make('Pendulum-v0')
policy = linear_policy.LinearPolicy(1, 3)
val = val_func_est.ValueFunction(0.90)

model = npg.NPG(policy, env, val)

model.train(K=50, N=3)

