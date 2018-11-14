""" An implementation of the natural policy gradient algorithm as introduced in https://arxiv.org/abs/1703.02660"""

import numpy as np
import torch
import linear_policy
import gym


class NPG:
    def __init__(self, policy, env):
        self.policy = policy
        self.env = env

    def train(self, K=1, N=0, T=0, gamma=0):
        """ Implementation of policy search with NPG
        K -- Number of iterations
        N -- umber of trajs
        T -- Number of time steps (?)
        gamma -- Discount factor"""

        # Init. policy params
        Theta = 0

        # Collect trajs by rolling out policy with Theta
        old_traj = self.rollout(N)

        for k in range(K):
            # Collect trajs by rolling out policy with Theta
            trajs = self.rollout(N)

            # Compute gradient for each (s,a) pair of the sampled trajectories
            Nabla_Theta = np.zeros((len(trajs), len(trajs[0])))
            for traj, t in zip(trajs, len(trajs)):
                for point, p in zip(traj, len(traj)):
                    grad = self.policy.get_gradient(point[0], point[1])
                    print(grad)
                    Nabla_Theta[t][p] = grad

            # Approx.value function
            V = conj_grad(old_traj)
            # Compute advantages
            A = self.gae(V, T, k, trajs[:][:][2])

            """
            # Compute policy gradient (2)
            # TODO: What is T? Maybe number of time steps?
            pg = self.pol_grad(Nabla_Theta, A, T)

            # Compute Fisher matrix (4)
            F = self.fish(Nabla_Theta, T)

            # Perform gradient ascent (5)
            # TODO: Normalize step size
            delta = 0
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

                observation, reward, done, info = env.step(action) # Take action

                point.append(reward)  # Save reward to tuple
                traj.append(point)  # Add Tuple to traj
            trajs.append(traj)

        return trajs

    def gae(self, V, T, k, R, gamma=1, lamb=0.1):
        """
        Estimates the advantage function using the GAE algorithm (https://arxiv.org/pdf/1506.02438.pdf)
        :param V: The estimated value function for the previous set of trajectories
        :param T: The number of time steps
        :param k: The current iteration (k=t)
        :param R: The rewards for the current trajectories
        :param gamma: Hyperparam.
        :param lamb: Hyperparam.
        :return: Estimated advantage function
        """
        A = 0
        for l in range(T):
            delta = R[k+l]+gamma*V[k+l+1]-V[k+l]
            A += ((gamma*lamb)**l)*delta
        return A

    def pol_grad(Nabla_Theta, A, T):
        """ Computes the policy gradient.
        Nabla_Theta -- Contains the log gradient for each state-action pair along trajs
        A -- Advantages based on trajs in current iteration
        T -- Number of time steps (?)"""
        exp_sum = 0
        for t in range(0, T):
            exp_sum = exp_sum + Nabla_Theta[t]*A[t]
        pg = exp_sum/T

        return pg


    def fish(self, Nabla_Theta, T):
        """ Computes the Fisher matrix.
        Nabla_Theta -- Log gragdient for each (s,a) pair
        T -- Number of time steps (?)"""
        F_sum = 0
        for t in range(0, T):
            F_sum = F_sum + Nabla_Theta[t] * Nabla_Theta[t].T
        F  = F_sum/T

        return F

    def grad_asc(self, Theta, pg, F, delta):
        """Performs gradient ascent on the parameter function
        Theta -- Params of the current policy
        pg -- Policy gradient
        F -- Fisher information matrix
        delta -- Normalized step size"""
        Theta = Theta + np.sqrt(delta/pg.T * (1/F) * pg) * (1/F) * pg

        return Theta

    def empi_re(self, states, N, T, gamma):
        """ Computes the empirical return for each state in each time step and every trajectory
        states -- The states in the current rollout
        N -- Number of trajs
        T -- Number of time steps (?)
        gamma -- Discount factor"""
        return 0

    def update_v_params(self, R):
        """ Updates the params of the value function in order to approximate it according to the empirical reward
        R -- Empirical reward"""
        return 0


env = gym.make('Pendulum-v0')
policy = linear_policy.LinearPolicy(1, 3)
model = NPG(policy, env)
model.train()
