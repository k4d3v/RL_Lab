"""
Use this file to check that your implementation complies with our evaluation
interface.
"""

import gym
from gym.wrappers.monitor import Monitor
from challenge1 import get_model, get_policy

# 1. Learn the model f: s, a -> s', r
env = Monitor(gym.make('Pendulum-v0'), 'training', video_callable=False, force=True)
env.seed(98251624)
max_num_samples = 10000
model = get_model(env, max_num_samples)
env.close()

# Your model will be tested on the quality of prediction
obs = env.reset()
act = env.action_space.sample()
nobs, rwd, _, _ = env.step(act)
nobs_pred, rwd_pred = model(obs, act)
print(f'truth = {nobs, rwd}\nmodel = {nobs_pred, rwd_pred}')

# 2. Perform dynamic programming using the learned model
env = Monitor(gym.make('Pendulum-v0'), 'evaluation')
env.seed(31186490)
policy = get_policy(model, env.observation_space, env.action_space)

# Your policy will be judged based on the average episode return
n_eval_episodes = 100
for _ in range(n_eval_episodes):
    done = False
    obs = env.reset()
    while not done:
        act = policy(obs)
        obs, _, done, _ = env.step(act)
av_ep_ret = sum(env.get_episode_rewards()) / len(env.get_episode_rewards())
print(f'average return per episode: {av_ep_ret}')
env.close()
