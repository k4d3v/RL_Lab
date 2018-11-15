""" An implementation of the natural policy gradient algorithm as introduced in https://arxiv.org/abs/1703.02660"""

import numpy as np
from numpy.linalg import inv
import torch
import linear_policy
import gym
import val_func_est


class NPG:
    def __init__(self, policy, env, val, log):
        self.policy = policy
        self.env = env
        self.val = val
        self.log = log

    def train(self, K=1, N=1, gamma=0.9, lamb=0.97):
        """ Implementation of policy search with NPG
        K -- Number of iterations
        N -- umber of trajs
        T -- Number of time steps (Trajectory length)
        gamma -- Discount factor"""

        for k in range(K):
            print("Iteration ", k)
            # Collect trajs by rolling out policy with Theta
            trajs = self.rollout(N)

            # Compute gradient for each (s,a) pair of the sampled trajectories
            log_prob_gradients = self.nabla_theta(trajs)

            # Compute Advantages
            vals = self.val.predict(trajs)
            adv = self.gae(trajs, vals, gamma, lamb)

            # Compute Vanilla Policy Gradient
            policy_gradient = self.pol_grad(log_prob_gradients, adv)

            # Compute Fisher Information Metric
            F = self.fish(log_prob_gradients)
            F_inv = inv(F)

            # Compute gradient ascent step (normalize step size * natural gradient)
            step = self.grad_asc_step(policy_gradient, F_inv)

            # Update params
            self.policy.update_params(step)

            # Update value function
            self.val.fit(trajs)

            print("--------------------")

    def rollout(self, N):
        """ Returns sampled trajs based on the stochastic policy
        N -- Number of trajectories"""
        trajs = []
        avg_reward=0.0
        for n in range(N):
            # Reset the environment
            observation = self.env.reset()
            episode_reward = 0.0
            done = False
            traj = []

            while not done:
                # env.render()
                point = []

                action = self.policy.get_action(torch.from_numpy(observation).view(3, 1).float())  # rollout policy

                point.append(torch.from_numpy(observation).view(3, 1).float())  # Save state to tuple
                point.append(action)  # Save action to tuple

                observation, reward, done, info = self.env.step(action) # Take action
                episode_reward += reward
                point.append(reward)  # Save reward to tuple
                traj.append(point)  # Add Tuple to traj

            avg_reward += episode_reward
            trajs.append(traj)

        print("Avg reward: ", (avg_reward/N).numpy())
        self.log.add((avg_reward/N).numpy()[0])
        return trajs

    def nabla_theta(self, trajs):
        """
        Computes the gradient for each state-action pair on the sampled trajectories
        :param trajs: Sampled trajectories
        :return: Estimated gradient
        """
        Nabla_Theta = []
        for traj in trajs:
            nabla_theta_traj = []
            for point in traj:
                grad = self.policy.get_gradient(point[0], point[1])
                nabla_theta_traj.append(grad)
            Nabla_Theta.append(nabla_theta_traj)
        return Nabla_Theta

    def pol_grad(self, log_prob_gradients, advantages):
        """ Computes the policy gradient.
        log_prob_gradients -- Contains the log gradient for each state-action pair along trajs
        advantages -- Advantages based on trajs in current iteration
        """
        policy_gradient = np.zeros((1, len(log_prob_gradients[0][0])))
        for i in range(len(log_prob_gradients)):
            traj_gradient = np.zeros((1, len(log_prob_gradients[i][0])))
            for j in range(len(log_prob_gradients[i])):
                traj_gradient += log_prob_gradients[i][j] * advantages[i][j]
            policy_gradient += traj_gradient / len(log_prob_gradients[i])
        return policy_gradient / len(log_prob_gradients)

    def fish(self, log_prob_gradients):
        """ Computes the Fisher matrix.
        log_prob_gradient -- Log gradient for each (s,a) pair
        """
        F = np.zeros((len(log_prob_gradients[0][0]), len(log_prob_gradients[0][0])))
        for i in range(len(log_prob_gradients)):
            F_traj = np.zeros((len(log_prob_gradients[0][0]), len(log_prob_gradients[0][0])))
            for j in range(len(log_prob_gradients[i])):
                F_traj += np.outer(log_prob_gradients[i][j], log_prob_gradients[i][j])
            F += F_traj / len(log_prob_gradients[i])

        return F / len(log_prob_gradients)

    def grad_asc_step(self, policy_gradient, F_inv, delta=0.05):
        """Performs gradient ascent on the parameter function
        pg -- Policy gradient
        F _inv-- Fisher information matrix inverse
        delta -- Normalized step size
        """
        alpha = np.sqrt(delta / (np.matmul(np.matmul(policy_gradient, F_inv), policy_gradient.T)))
        nat_grad = np.matmul(F_inv, policy_gradient.T)
        return alpha*nat_grad

    def gae(self, trajs, vals, gamma, lamb):
        """
        Estimates the advantage function using the GAE algorithm (https://arxiv.org/pdf/1506.02438.pdf)
        :param trajs: The trajectories
        :param vals: The estimated value function for the previous set of trajectories
        :param gamma: Hyperparam.
        :param lamb: Hyperparam.
        :return: Estimated advantage function
        """
        all_advantages=[]
        for i in range(len(trajs)):
            traj_advantages=[]
            for j in range(len(trajs[i])):
                adv = 0.0
                for l in range(0, len(trajs[i])-j-1):
                    delta = trajs[i][j+l][2].numpy() + gamma*vals[i][j+l+1] - vals[i][j+l]
                    adv += ((gamma*lamb)**l)*delta

                traj_advantages.append(adv)
            all_advantages.append(traj_advantages)

        return all_advantages


