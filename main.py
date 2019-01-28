""" Main file for testing the PILCO implementation."""
import random

from pilco import PILCO

random.seed(42)

env_name = 'CartpoleStabShort-v0'

agent = PILCO(env_name)
optimal_policy = agent.train()
