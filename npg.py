""" An implementation of the natural policy gradient algorithm as introduced in https://arxiv.org/abs/1703.02660"""

import numpy as np
from numpy.linalg import inv
import torch
import linear_policy
import gym
import val_func_est


class NPG:
    def __init__(self, policy, env, val):
        self.policy = policy
        self.env = env
        self.val = val

    def train(self, K=1, N=1, gamma=0):
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
            # TODO

            # Compute Vanilla Policy Gradient
            policy_gradient = self.pol_grad(log_prob_gradients, 0)

            # Compute Fisher Information Metric
            F = self.fish(log_prob_gradients)
            F_inv = inv(F)

            # Compute gradient ascent step
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
                traj_gradient += log_prob_gradients[i][j] * 1  # Set all Advantages to 1
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

    def gae(self, V, T, R, gamma=1, lamb=0.1):
        """
        Estimates the advantage function using the GAE algorithm (https://arxiv.org/pdf/1506.02438.pdf)
        :param V: The estimated value function for the previous set of trajectories
        :param T: The number of time steps
        :param R: The rewards for the current trajectories
        :param gamma: Hyperparam.
        :param lamb: Hyperparam.
        :return: Estimated advantage function
        """
        A = []
        for n in range(len(R)):
            curr_A = []
            for t in range(T):
                A_t = 0
                for l in range(T):
                    delta = R[t + l] + gamma * V[t + l + 1] - V[t + l]
                    A_t += ((gamma * lamb) ** l) * delta
                curr_A.append(A_t)
            A.append(curr_A)
        return A
