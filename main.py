""" Main file for testing the implementation of the NPG algorithm"""

import npg
import gym
import linear_policy


env = gym.make('Pendulum-v0')
policy = linear_policy.LinearPolicy(1, 3)

model = npg.NPG(policy, env)

trajs = model.rollout(10)

Nabla_Theta = model.nabla_theta(trajs)

print("some comment")