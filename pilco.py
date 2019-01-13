""" An implementation of the PILCO algorithm as shown in
http://www.icml-2011.org/papers/323_icmlpaper.pdf"""
import gym
import quanser_robots
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
        """
        Perform PILCO algorithm on initial random RBF policy
        :return: Optimal policy
        """

        # Init. environment
        env = gym.make(self.env_name)
        # Dimension of states
        s_dim = env.observation_space.shape[0]

        # Initial J random rollouts
        data = []
        old_Theta = Policy(env)
        for j in range(self.J):
            # Sample controller params
            Theta = Policy(env)

            # Apply random control signals and record data
            data.append(Theta.rollout())

        # Learn hyperparams for dynamics GP
        dyn_model = DynModel(s_dim, data)
        lambs = dyn_model.estimate_hyperparams()

        # Controlled learning (N iterations)
        for n in range(self.N):
            print("Round ", n)

            # Learn GP dynamics model using all data (Sec. 2.1)
            dyn_model = DynModel(s_dim, data, lambs) # TODO: Impl. dyn. model

            """ For testing the dyn model accuracy
            s = np.concatenate((data[0][0][0], data[0][0][1]))
            m, sig = dyn_model.predict(s)
            """

            i = 0
            while True:
                print("Policy search iteration ", i)

                mu_delta, Sigma_delta = self.approximate_p_delta_t() # TODO

                # Approx. inference for policy evaluation (Sec. 2.2)
                # Get J^pi(Theta) (10-12), (24)
                J = self.get_J(mu_delta, Sigma_delta, dyn_model)  # TODO

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

        return Theta

    def get_J(self, mu_delta, Sigma_delta, dyn_model):
        """
        Constructs a gaussian approximation for every p(x_t) based on subsequent one-step predictions and computes the expected values
        :param mu_delta: Mean of approximated p_delta_t
        :param Sigma_delta: Std of approximated p_delta_t
        :param dyn_model: Trained dynamics model
        :return: J (Expected values)
        """
        # Construct gaussian approximation of p(x_t)
        for t in range(dyn_model.big_t):
            mu_t = dyn_model.mu+mu_delta
            Sigma_t = 0
            acov = 0

        # Compute the expected values
        E_x_t = 0

        return E_x_t

    def get_dJ(self, J):
        return 0

    def approximate_p_delta_t(self):
        return 0, 0
