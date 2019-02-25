import numpy as np
import torch


class Evaluator:
    """
    A class for evaluating a learnt policy
    """

    def __init__(self, policy, env):
        """
        Init.
        :param policy: The current policy
        :param env: Current environment
        """
        self.policy = policy
        self.env = env

        # State dimension
        self.s_dim = self.env.observation_space.shape[0]

    def evaluate(self, n, render=False, limit=False):
        """
        Collect rewards based on actions sampled from the learnt policy
        :param n: Number of rollouts
        :return: Average reward
        """
        avg_reward = 0.0
        min_reward = 100000
        max_reward = -100000

        #acts = []
        for _ in range(n):

            # Reset the environment
            observation = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                if render:
                    self.env.render()

                action = self.policy.get_action(torch.Tensor(observation).view(self.s_dim, 1)) \
                    if self.env.spec.id == "BallBalancerSim-v0" \
                    else np.clip(self.policy.get_action(torch.Tensor(observation).view(self.s_dim, 1)), -6, 6)
                #action = self.policy.get_action(torch.Tensor(observation).view(self.s_dim, 1))
                #acts.append(action)
                #print(action)
                observation, reward, done, _ = self.env.step(action)  # Take action

                if limit:
                    min_s0, max_s0 = self.env.observation_space.low[0], self.env.observation_space.high[0]
                    distl = np.abs(observation[0]-min_s0)
                    distr = np.abs(observation[0]-max_s0)
                    print(distl)
                    print(distr)
                    if distl<0.1 or distr<0.1:
                        break

                episode_reward += reward

            if episode_reward > max_reward:
                max_reward = episode_reward
            if episode_reward < min_reward:
                min_reward = episode_reward
            avg_reward += episode_reward

        #amin, amax = np.min(acts), np.max(acts)
        return avg_reward / n
