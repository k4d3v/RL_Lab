""" Main file for testing the PILCO implementation."""
import numpy as np
import pickle

from pilco_lin import PILCO

np.random.seed(12)

env_name = 'CartpoleStabShort-v0'

agent = PILCO(env_name)
optimal_policy = agent.train()

for _ in range(100):
    optimal_policy.rollout(render=True)

pickle.dump(optimal_policy, open(env_name+".p", "wb"))