""" An implementation of the PILCO algorithm as shown in
http://www.doc.ic.ac.uk/~mpd37/publications/pami_final_w_appendix.pdf """

import numpy as np

from opt_params import Opt
from policy import Policy
from dyn_model import DynModel


class PILCO:
    def __init__(self, J=1, N=1):
        """
        :param J: Number of rollouts
        :param N: Number of iterations
        """
        self.J = J
        self.N = N

    def settings(self):
        """
        Set initial values
        gauss: Gaussian distribution
        policy: policy structure
        H: rollout horizon in steps
        plant: the dynamical system structure
        cost: cost structure
        """
        gauss = 0
        policy = Policy()
        H = 0
        plant = 0
        cost = 0
        return gauss, policy, H, plant, cost

    def train(self):
        # Load settings
        gauss, policy, H, plant, cost = self.settings()

        # Set other params
        opt = Opt()

        # Initial J random rollouts
        for j in range(self.J):
            # Sample controller params
            Theta = np.random.normal(size=(self.N, self.N))

            # Apply random control signals and record data
            x, y, real_cost, latent = self.rollout(gauss, policy, H, plant, cost)

        # Controlled learning (N iterations)
        for n in range(self.N):
            # Learn GP dynamics model using all data (Sec. 3.1)
            dyno, dyni, difi, train_opt = 0, 0, 0, 0
            dyn_model = self.trainDynModel(policy, plant, dyno, dyni, difi, train_opt)

            # Approx. inference for policy evaluation (Sec. 3.2)

            # Get J^pi(Theta) (9), (10), (11)

            # Policy improvement based on the gradient (Sec. 3.3)
            # Get the gradient of J (12)-(16)

            # Learn policy
            mu0Sim, S0Sim = 0,0,
            self.learnPolicy(policy, opt, mu0Sim, S0Sim, dyn_model, plant, cost, H, n)

            # Update Theta (CG or L-BFGS)

            # Convergence check

            # Set new optimal policy

            # Apply policy to system and record
            self.applyController()

            # Convergence check

    def rollout(self, gauss, policy, H, plant, cost):
        """
        Rolls out a policy
        :param gauss: Normal distribution
        :param policy: Current policy
        :param H: Rollout horizon in steps
        :param plant: Dynamics system
        :param cost: Cost function
        :return: x, y, L, latent
        #         x: matrix of observed states
        #         y: matrix of corresponding observed successor states
        #         L: Real cost incurred at each time step
        #         latent: matrix of latent states
        """
        x, y, L, latent = 0, 0, 0, 0
        return x, y, L, latent

    def trainDynModel(self, policy, plant, dyno, dyni, difi, train_opt):
        """Trains GP dynamics model"""
        dyn_model = DynModel()
        return dyn_model


    def learnPolicy(self, policy, opt, mu0Sim, S0Sim, dyn_model, plant, cost, H, n):
        """
        Learns a policy
        :param policy:
        :param opt:
        :param mu0Sim:
        :param S0Sim:
        :param dyn_model:
        :param plant:
        :param cost:
        :param H:
        :param n: Current iteration
        :return:
        """
        # 1. Update the policy
        opt.fh = 1
        policy.p, fX3 = self.minimize(policy.p, 'value', opt, mu0Sim, S0Sim, dyn_model, policy, plant, cost, H)

        # TODO: Plot overall optimization progress

        # 2. Predict trajectory from p(x0) and compute cost trajectory
        M[n], Sigma[n] = pred(policy, plant, dyn_model, mu0Sim[:, 1], S0Sim, H)

        # Predict cost trajectory
        fantasy.mean[n], fantasy.std[n] = calcCost(cost, M[n], Sigma[n])

        # TODO: Plot predicted immediate costs (as a function of the time steps )

        return 0

    def applyController(self):
        """
        Applies the learned controller to a (simulated) system
        :return:
        """
        # 1.Generate trajectory rollout given the current policy

        # 2. Make many rollouts to test the controller quality

        # 3. Save data
        return 0

    def minimize(self, p, val, opt, mu0Sim, S0Sim, dyn_model, policy, plant, cost, H):
        """
        Learns a policy
        :param p:
        :param val:
        :param opt:
        :param mu0Sim:
        :param S0Sim:
        :param dyn_model:
        :param policy:
        :param plant:
        :param cost:
        :param H:
        :return:
        """
        p = 0
        fX3 = 0
        return p, fX3
