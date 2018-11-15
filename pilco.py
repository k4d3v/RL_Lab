""" An implementation of the PILCO algorithm as shown in http://www.doc.ic.ac.uk/~mpd37/publications/pami_final_w_appendix.pdf"""

import numpy as np


class PILCO:
    def __init__(self, J=1, N=1):
        """
        :param J: Number of rollouts
        :param N: Number of iterations
        """
        self.J = J
        self.N = N

    def train(self):
        # Initial J random rollouts
        for j in range(self.J):
            # Sample controller params
            Theta = np.random.normal(size=(self.N, self.N))

            # Apply random control signals and record data
            x, y, L, latent = self.rollout(start, policy, H, plant, cost)

        # Controlled learning (N iterations)
        for n in range(self.N):
            # Learn GP dynamics model using all data (Sec. 3.1)
            self.trainDynModel()

            # Approx. inference for policy evaluation (Sec. 3.2)

            # Get J^pi(Theta) (9), (10), (11)

            # Policy improvement based on the gradient (Sec. 3.3)
            # Get the gradient of J (12)-(16)

            # Learn policy
            self.learnPolicy()
            # Update Theta (CG or L-BFGS)

            # Convergence check


            # Set new optimal policy

            # Apply policy to system and record
            self.applyController()
            # Convergence check

    def rollout(self, start, policy, H, plant, cost):
        # """
        # :param start: vector containing initial states (without controls)
        # :param policy: policy structure
        # :param H: rollout horizon in steps
        # :param plant: the dynamical system structure
        # :param cost: cost structure
        # :return: x, y, L, latent
        #         x: matrix of observed states
        #         y: matrix of corresponding observed successor states
        #         L: cost incurred at each time step
        #         latent: matrix of latent states
        # """

        # return x, y, L, latent
        return 0
    # Train (GP) dynamics model
    def trainDynModel(self):
        # Train GP dynamics model
        return 0

    # Learn Policy
    def learnPolicy(self):
        # 1. Updata the policy

        # 2. Predict trajectory from p(x0) and compute cost trajectory
        return 0

    # Apply the learned controller to a (simulated) system
    def applyController(self):
        # 1.Generate trajectory rollout given the current policy

        # 2. Make many rollouts to test the controller quality

        # 3. Save data
        return 0
