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

        while True:
            try:
                # Initial J random rollouts
                data = []
                old_policy, policy = Policy(env), Policy(env)
                for j in range(self.J):
                    # Sample controller params
                    policy = Policy(env)

                    # Apply random control signals and record data
                    data.append(policy.rollout())

                dyn_model = DynModel(s_dim, data)
                print("Average GP error before optimizing the hyperparams: ", dyn_model.training_error())

                # Learn hyperparams for dynamics GP
                lambs = dyn_model.estimate_hyperparams()
                break
            except ValueError:
                print("Oops, some stupid numerical problem. Trying again...")

        # Controlled learning (N iterations)
        for n in range(self.N):
            print("Round ", n)

            # Learn GP dynamics model using all data (Sec. 2.1)
            dyn_model = DynModel(s_dim, data, lambs)
            print("Average GP error: ", dyn_model.training_error())

            i = 0
            while True:
                print("Policy search iteration ", i)

                mu_delta, Sigma_delta = self.approximate_p_delta_t(dyn_model) # TODO

                # Approx. inference for policy evaluation (Sec. 2.2)
                # Get J^pi(policy) (10-12), (24)
                J = self.get_J(mu_delta, Sigma_delta, dyn_model)  # TODO

                # Policy improvement based on the gradient (Sec. 2.3)
                # Get the gradient of J (26-30)
                # TODO: Torch gradient
                dJ = self.get_dJ(J)

                # Learn policy
                # Update policy (CG or L-BFGS)
                policy.update(J, dJ)  # TODO

                # Convergence check
                # TODO
                if policy.check_convergence(old_policy):
                    break

                old_policy = policy
                i += 1

            # Apply new optimal policy to system (One episode) and record
            new_data = policy.rollout()
            data.append(new_data)

            # Convergence check
            # TODO

        return policy

    def get_J(self, mu_delta, Sigma_delta, dyn_model):
        """
        Returns a function which constructs a gaussian approximation for every p(x_t) based on subsequent one-step predictions and computes the expected values
        :param mu_delta: Mean of approximated p_delta_t
        :param Sigma_delta: Std of approximated p_delta_t
        :param dyn_model: Trained dynamics model
        :return: Function for estimating J (Expected values)
        """
        def J():
            # Construct gaussian approximation of p(x_t)
            for t in range(len(dyn_model.x)):
                mu_t = dyn_model.mu+mu_delta
                Sigma_t = 0
                acov = 0

            # Compute the expected values
            E_x_t = 0

            return E_x_t
        return J

    def get_dJ(self, J):
        """
        Returns a function which can estimate the gradient of the expected return
        :return: function dJ
        """
        def dJ():
            return 0
        return dJ

    def approximate_p_delta_t(self, dyn_model):
        # calculate   mu_delta
        # init
        n = len(dyn_model.x)     # input
        D = 5      # s_dim
        x_s = dyn_model.x
        Sigma = dyn_model.sigma
        mu_s_t_1, Sigma_t_1= dyn_model.predict(dyn_model.x)    # mu_schlange(t-1) is the mean of the "test" input distribution p(x[t-1],u[t-1])
        q = np.zeros(D, n)
        y = np.zeros(n, D)   # output
        v = np.zeros(1, n)
        length_scale = []    ### nur fuer test, noch nicht bestimmt
        Lambda = np.zeros(D, D)
        alpha = np.ones(1, D)    ### nur fuer test, noch nicht bestimmt

        # calculate Lambda, under (6)
        for i in range(1, D + 1):
            Lambda[i][i]=length_scale[i]**2

        # calculate   q_ai
        for a in range(1, D + 1):
            for i in range(1, n + 1):
                # (16)
                v[i] = x_s[i] - mu_s_t_1  # x_schlange and mu_schlange in paper, x_schlange is training input,
                                             # mu_schlange is the mean of the "test" input distribution p(x[t-1],u[t-1])
                # (15)
                q[a][i] = alpha[a] ** 2 / np.sqrt(np.abs(Sigma[t - 1] * np.linalg.inv(Lambda[a]) + np.eye(D))) * np.exp(
                    (-1 / 2) * v[i].T * np.linalg.inv(Sigma[t - 1] + Lambda[a]) * v[i])   # Sigma[t-1] is variances at time t-1 from GP

        # calculate   K
        K = np.zeros(1, D) # K for all input und dimension, each term is also a matrix for all input and output
        K_dim = np.zeros(n, n) # K_dim tmp to save the result of every dimension
        for a in range(1, D+1):
            for i in range(1, n+1):
                for j in range(1, n+1):
                    # (6)
                    K_dim[i][j] = alpha**2 *np.exp(-0.5*(x_s[i]-x_s[j]).T *np.linalg.inv(Lambda)*(x_s[i]-x_s[j])) # x_s is x_schlange in paper,training input
            K[1][a] = K_dim


        # calculate   beta, under (14)
        beta = np.zeros(n, D)
        for a in range(1, D + 1):
            beta[:, a] = np.linalg.inv(K[1][a]) * y[:, a]

        # calculate   mu_delta (14)
        mu_delta = []
        for a in range(1, D + 1):
            mu_delta[a] = beta[:, a].T * q[a, :]

        # calculate   k
        k = np.zeros(1, D)  # k for all input und dimension, each term is also a matrix for all input and output
        k_dim = np.zeros(n, n)  # k_dim tmp to save the result of every dimension
        for a in range(1, D + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    # (6)
                    k_dim[i][j] = alpha ** 2 * np.exp(-0.5 * (x_s[i] - mu_s_t_1).T * np.linalg.inv(Lambda[a][a]) * (x_s[i] - mu_s_t_1))  # x_s is x_schlange in paper,training input
            k[1][a] = k_dim



        # calculate   Sigma_delta
        # init
        z = np.zeros(n, n)
        R = np.zeros(D, D)
        Sigma_delta = []

        for i in range(1, n + 1):
            for j in range(1, n + 1):
                for a in range(1, D + 1):
                    for b in range(1, D + 1):
                        # under (22)
                        z[i][j] = np.linalg.inv(Lambda[a][a]) * v[i] + np.linalg.inv(Lambda[b][b]) * v[j]
                        R[a][b] = Sigma[t - 1] * (np.linalg.inv(Lambda[a][a]) + np.linalg.inv(Lambda[b][b])) + np.eye(D)
                        if a != b:
                            # (18)=(20)=beta[a].T*(22)*beta[b]
                            Sigma_delta[a][b] = beta[:, a].T * (((k[1][a] * k[1][b]) / np.sqrt(abs(R[a][b]))) * np.exp(0.5 * z[i][j].T * np.linalg.inv(R[a][b] * Sigma[t - 1] * z[i][j]))) * beta[:, b] - mu_delta[a] * mu_delta[b]
                        else:
                            # (17)=(23)+(20)
                            Sigma_delta[a][a] = alpha[a] ** 2 -np.trace(np.linalg.inv(K[1][a] + np.eye(D))) * (((k[1][a] * k[1][a]) / np.sqrt(np.abs(R[a][a]))) * np.exp(0.5 * z[i][j].T * np.linalg.inv(R[a][a] * Sigma_t_1 * z[i][j]))) + beta[:,a].T * (((k[1][a] * k[1][a]) / np.sqrt(np.abs(R[a][a]))) * np.exp(0.5 * z[i][j].T * np.linalg.inv(R[a][a] * Sigma_t_1 * z[i][j]))) * beta[:, a] - mu_delta[a] * mu_delta[a]

        return mu_delta, Sigma_delta
