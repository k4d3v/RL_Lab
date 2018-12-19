""" Main file for testing the PILCO implementation."""

from pilco import PILCO

env_name = 'CartpoleStabShort-v0'

agent = PILCO(env_name)
optimal_policy = agent.train()
