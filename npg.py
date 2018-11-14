""" An implementation of the natural policy gradient algorithm as introduced in https://arxiv.org/abs/1703.02660"""

import numpy as np
import torch
import linear_policy
import gym
import val_func_est


class NPG:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def train(self, K=1, N=1, T=100, gamma=0):
        """ Implementation of policy search with NPG
        K -- Number of iterations
        N -- umber of trajs
        T -- Number of time steps (Trajectory length)
        gamma -- Discount factor"""

        # Collect trajs by rolling out policy with Theta
        old_trajs = self.rollout(N)

        for k in range(K):
            # Collect trajs by rolling out policy with Theta
            trajs = self.rollout(N)

            # Compute gradient for each (s,a) pair of the sampled trajectories
            Nabla_Theta = self.nabla_theta(trajs)

            """
            # Approx.value function
            V = val_func_est.conj_grad(old_trajs)
            # Compute advantages
            A = self.gae(V, T, trajs[:][:][2])
            
            # Compute policy gradient (2)
            pg = self.pol_grad(Nabla_Theta, A, T)

            # Compute Fisher matrix (4)
            F = self.fish(Nabla_Theta, T)

            # Perform gradient ascent (5)
            # TODO: Normalize step size
            Theta = self.grad_asc(Theta, pg, F, delta)

            # Update params of value function in order to approx. V(s_t^n)
            R = self.empi_re(states, N, T, gamma)
            v_params = self.update_v_params(R)

            # Return params of optimal policy
            return Theta
            """

    def rollout(self, N):
        """ Returns sampled trajs based on the stochastic policy
        N -- Number of trajectories"""
        trajs = []
        for n in range(N):
            # Reset the environment
            observation = self.env.reset()
            done = False
            traj = []

            while not done:
                # env.render()
                point = []

                action = self.policy.get_action(torch.from_numpy(observation).view(3, 1).float())  # rollout policy

                point.append(torch.from_numpy(observation).view(3, 1).float())  # Save state to tuple
                point.append(action)  # Save action to tuple

                observation, reward, done, info = self.env.step(action) # Take action

                point.append(reward)  # Save reward to tuple
                traj.append(point)  # Add Tuple to traj
            trajs.append(traj)

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
                    delta = R[t+l]+gamma*V[t+l+1]-V[t+l]
                    A_t += ((gamma*lamb)**l)*delta
                curr_A.append(A_t)
            A.append(curr_A)
        return A

    def pol_grad(self, Nabla_Theta, A, T):
        """ Computes the policy gradient.
        Nabla_Theta -- Contains the log gradient for each state-action pair along trajs
        A -- Advantages based on trajs in current iteration
        T -- Trajectory length"""
        pg = []
        for n in range(len(A)):
            exp_sum = 0
            for t in range(T):
                exp_sum += Nabla_Theta[n][t]*A[n][t]
            pg.append(exp_sum/T)

        return pg


    def fish(self, Nabla_Theta, T):
        """ Computes the Fisher matrix.
        Nabla_Theta -- Log gradient for each (s,a) pair
        T -- Trajectory length"""
        F = []
        for n in range(len(Nabla_Theta)):
            F_sum = 0
            for t in range(T):
                F_sum +=  Nabla_Theta[n][t] * Nabla_Theta[n][t].T
            F.append(F_sum/T)

        return F

    def grad_asc(self, Theta, pg, F, delta=0.05):
        """Performs gradient ascent on the parameter function
        Theta -- Params of the current policy
        pg -- Policy gradient
        F -- Fisher information matrix
        delta -- Normalized step size"""
        for n in range(len(F)):
            Theta[n] += np.sqrt(delta/pg[n].T * (1/F[n]) * pg[n]) * (1/F[n]) * pg[n]

        return Theta

    def empi_re(self, states, N, T, gamma):
        """ Computes the empirical return for each state in each time step and every trajectory
        states -- The states in the current rollout
        N -- Number of trajs
        T -- Trajectory length
        gamma -- Discount factor"""
        return 0

    def update_v_params(self, R):
        """ Updates the params of the value function in order to approximate it according to the empirical reward
        R -- Empirical reward"""
        V = R
        return V