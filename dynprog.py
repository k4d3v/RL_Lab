from value_func import FitNN
import torch
import numpy as np


class ValIter:
    def __init__(self, policy, env, reward, dynamics):
        self.policy = policy
        self.env = env
        self.reward = reward
        self.dynamics = dynamics

    def train(self, n_samples=10000, discount=0.1):
        # Generate Points by rolling out the policy
        print("1. Rollout policy...")
        points = self.rollout(n_samples)

        # Prepare the points for the NN
        x, y, y1 = [], [], []
        for point in points:
            old, new, act, rew = point
            x.append(np.append(old, act))
            y.append(rew)
            y1.append(new)
        x = torch.Tensor(x)
        y = torch.Tensor(y).view(len(y), 1)
        y1 = torch.Tensor(y1)

        # Fit reward dunction
        print("-----------------------------------------------------")
        print("2. Fit reward Function... This will take around 30 sec")
        self.reward.fit_batch(x, y, 1000, 256)

        # Fit dynamics function
        print("-------------------------------------------------------")
        print("3. Fit dynamics Function... This will take around 30 sec")
        self.dynamics.fit_batch(x, y1, 1000, 256)

        # Perform DP with Value Iteration / Policy Iteration
        # TODO

        # Compute cumulative reward over 100 episodes
        # TODO

        # TODO: State space discretization

        # TODO: Learn dynamics (next state according to an action) by training the NN

        """ TODO: Learn reward matrix for the discretized state space 
        by sampling randomly according to the initial exploration policy"""
        R = np.zeros((1,1))
        x,y = R.shape[0], R.shape[1]

        # TODO: Plot accuracy of the model for different numbers of samples

        # TODO: Difference Value and Policy Iteration?

        # Init. episode
        init_state = self.env.reset()

        actions = [self.policy.getAction(init_state) for i in range(n_samples)]

        policy = np.full((x, y), 0)

        # Init
        Vk = np.zeros((x, y))

        # Repeat
        while True:
            # Compute Q function
            Vk_new = np.zeros((x, y))
            for i in range(x):
                for j in range(y):

                    currQ = []
                    for a in actions:
                        currQ.append(R[i][j] + discount * self.calc_reward(Vk, [i, j], a))

                    # Compute V function
                    Vk_new[i][j] = self.max_action(currQ, R, 0)

                    # Update policy
                    policy[i][j] = self.find_policy(currQ)

            # Check convergence
            if (Vk == Vk_new).all():
                break
            Vk = Vk_new

        """
        Compute the cumulative reward over 100 episodes
        Compare results(plots and total reward) for different discretizations
        Plot best results of Value function and Policy for Pendulum - v0
        """

        return Vk, policy

    def max_action(self, V, R, discount):
        """Computes the V function. That is the maximum of the V parameter (Current possible rewards)
        """
        return max(V)

    def find_policy(self, V):
        """Finds an optimal policy for a state for 15 time steps, given the V function
        """
        return np.argmax(V)

    def calc_reward(self, currVt,s,a):
        """Calculates a Q value for a given state-action pair
        """
        i = s[0]+a[0]
        j = s[1]+a[1]
        # Return 0 if action is impossible and reward if not
        return currVt[i][j]

    def rollout(self, num):
        """
        Rolls out the policy for num timesteps
        """
        points = []
        points_collected = 0
        while points_collected < num:

            # Reset the environment
            observation = self.env.reset()
            done = False

            while not done and points_collected < num:
                old_observation = observation
                action = self.env.action_space.sample()
                observation, reward, done, _ = self.env.step(action)  # Take action
                points.append([old_observation, observation, action, reward])
                points_collected += 1

        return points




