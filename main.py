""" Main file for testing the PILCO implementation.
pilco_lin uses a linear policy.
pilco uses an RBF policy without predicting the long-term state distribution.
"""
import numpy as np
import pickle

from pilco_lin import PILCO
#from pilco import PILCO

np.random.seed(12)

#env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0',
#             'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']
env_names = ['CartpoleStabShort-v0']

for env_name in env_names:
    # Train agent
    agent = PILCO(env_name)
    optimal_policy = agent.train()

    # Run learnt policy
    for _ in range(100):
        optimal_policy.rollout(render=True)

    # Save policy params
    pickle.dump(optimal_policy.param_array(), open(env_name+".p", "wb"))
