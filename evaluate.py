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

    def evaluate(self, n):
        """
        Collect rewards based on actions sampled from the learnt policy
        :param n: Number of rollouts
        :return: Average reward
        """
        avg_reward = 0.0
        min_reward = 100000
        max_reward = -100000

        for _ in range(n):

            # Reset the environment
            observation = self.env.reset()
            episode_reward = 0.0
            done = False

            while not done:
                # env.render()

                action = self.policy.get_action(torch.Tensor(observation).view(5, 1))

                observation, reward, done, _ = self.env.step(action)  # Take action

                episode_reward += reward

            if episode_reward > max_reward:
                max_reward = episode_reward
            if episode_reward < min_reward:
                min_reward = episode_reward
            avg_reward += episode_reward

        return avg_reward / n
