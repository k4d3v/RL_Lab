from train_sim import train


""" High level script for training a NPG agent on different simulated environments with varying hyperparameters.
The plots and learnt policies are saved."""

#env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0',
#             'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']
env_names = ['CartpoleStabShort-v0']

# Train with delta (learning rate) as hyperparam
train(env_names, True)
# Train with num of trajs as hyperparam
train(env_names)