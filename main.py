""" Main file for testing the implementation of the NPG algorithm"""

import npg
import gym
import linear_policy
import val_func_est
import logger


env = gym.make('Pendulum-v0')
policy = linear_policy.LinearPolicy(1, 3)
val = val_func_est.ValueFunction(0.90)
log = logger.Logger()

model = npg.NPG(policy, env, val, log)
model.train(K=200, N=1)

log.plot()
