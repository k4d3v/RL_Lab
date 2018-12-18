""" An implementation of the PILCO algorithm as shown in
http://www.icml-2011.org/papers/323_icmlpaper.pdf"""
import gym
import numpy as np

from policy import Policy
from dyn_model import DynModel


class PILCO:
    def __init__(self, env_name, J=1, N=1):
        """
        :param env_name: Name of the environment
        :param J: Number of rollouts
        :param N: Number of iterations
        """
        self.env_name = env_name
        self.J = J
        self.N = N

    def train(self):

        # Init. environment
        env = gym.make(self.env_name)

        # Initial J random rollouts
        data = []
        # Sample controller params
        Theta = Policy()
        for j in range(self.J):
            # Apply random control signals and record data
            data.append(Theta.rollout())  # TODO: Impl. rollout

            # Sample controller params
            Theta = Policy()

        # Controlled learning (N iterations)
        old_Theta = Policy()
        for n in range(self.N):
            print("Round ", n)

            # Learn GP dynamics model using all data (Sec. 2.1)
            dyn_model = DynModel()
            dyn_model.train(data)  # TODO: Impl. dyn. model

            i = 0
            while True:
                print("Policy search iteration ", i)
                # Approx. inference for policy evaluation (Sec. 2.2)
                # Get J^pi(Theta) (10-12), (24)
                J = self.get_J()  # TODO

                # Policy improvement based on the gradient (Sec. 2.3)
                # Get the gradient of J (26-30)
                # TODO: Torch gradient
                dJ = self.get_dJ(J)

                # Learn policy
                # Update Theta (CG or L-BFGS)
                Theta.update(dJ)  # TODO

                # Convergence check
                # TODO
                if Theta.check_convergence(old_Theta):
                    break

                old_Theta = Theta
                i += 1

            # Apply new optimal policy to system (One episode) and record
            new_data = Theta.rollout()
            data.append(new_data)

            # Convergence check
            # TODO

    def get_J(self):
        return 0

    def get_dJ(self, J):
        return 0
