import pickle
import gym
import quanser_robots

from linear_policy import LinearPolicy as Policy


env_name = 'CartpoleStabShort-v0'
pol = Policy(gym.make(env_name))
par = pickle.load(open(env_name+".p", "rb"))
pol.assign_Theta(par)

for _ in range(100):
    pol.rollout(render=True)
