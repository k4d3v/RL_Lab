""" Main file for testing the implementation of the NPG algorithm"""

import npg
import gym
import linear_policy
import val_func_est
import logger
import quanser_robots

env = gym.make('CartpoleStabShort-v0')
policy = linear_policy.LinearPolicy(1, 5)
val = val_func_est.ValueFunction()
log = logger.Logger()

model = npg.NPG(policy, env, val, log)
model.train(K=500, N=5)

log.plot()
