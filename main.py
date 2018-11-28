""" Main file for testing the implementation for the challenge."""

import gym
import quanser_robots

from dyn_prog import DynProg
from policy import RandomExplorationPolicy
from fun_approximator import FitNN

# Solve for Pendulum
#env = gym.make("Pendulum-v0")
env = gym.make("Pendulum-v2")
policy = RandomExplorationPolicy()

# Dimension of states
s_dim = env.reset().shape[0]
reward = FitNN(s_dim+1, 1)
dynamics = FitNN(s_dim+1, s_dim)

print("Rollout policy...")
points = reward.rollout(100, env)

reward.learn(False, points)
dynamics.learn(True, points)
# TODO: Plot accuracy of the model for different numbers of samples

agent = DynProg(policy, env, reward, dynamics)
Vk, pol = agent.train_val_iter()
Vk, pol = agent.train_pol_iter()
# TODO: Compare results(plots and total reward) for different discretizations
# TODO: Plot best results of Value function and Policy for Pendulum - v2

# TODO: Swing up on Qube-v0
