""" Main file for testing the PILCO implementation."""
import numpy as np

from pilco_lin import PILCO

np.random.seed(12)

env_name = 'CartpoleStabShort-v0'

agent = PILCO(env_name)
optimal_policy = agent.train()
