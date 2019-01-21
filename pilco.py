""" An implementation of the PILCO algorithm as shown in
http://www.icml-2011.org/papers/323_icmlpaper.pdf"""
import gym
import quanser_robots
import numpy as np
from torch.autograd import grad

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

                # dyn_model = DynModel(s_dim, data)
                # print("Average GP error before optimizing the hyperparams: ", dyn_model.training_error())

                # Learn hyperparams for dynamics GP
                # lambs = dyn_model.estimate_hyperparams()
                break
            except ValueError:
                print("Oops, some stupid numerical problem. Trying again...")

        # Controlled learning (N iterations)
        for n in range(self.N):
            print("Round ", n)

            # Learn GP dynamics model using all data (Sec. 2.1)
            #lambs = [1]*(s_dim+1)
            #dyn_model = DynModel(s_dim, data, lambs)
            #print("Average GP error: ", dyn_model.training_error())
            dyn_model = DynModel(s_dim, data)
            print("Average GP error: ", dyn_model.training_error_gp())

            i = 0
            while True:
                print("Policy search iteration ", i)

                mu_delta, Sigma_delta = self.approximate_p_delta_t(dyn_model, policy)  # TODO
                # TODO: Maybe cascade predictions for n time steps

                # Approx. inference for policy evaluation (Sec. 2.2)
                # Get J^pi(policy) (10-12), (24)
                J = self.get_J(mu_delta, Sigma_delta, dyn_model)  # TODO

                # Policy improvement based on the gradient (Sec. 2.3)
                # Get the gradient of J (26-30)
                # TODO: Torch gradient
                dJ = self.get_dJ(policy.Theta)

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

        def E_x_t(x):
            N = dyn_model.N  # number of inputs
            x_t = 0  # input at t
            x_t_1 = 0  # input at t-1

            mu_t_1, sigma_t_1 = dyn_model.predict(dyn_model.x)  # ?

            x_target = 0  # target state
            sigma_c = 0  # control the width of cost function, top right of page 4

            # under 2.1
            delta_t = x_t - x_t_1

            # (10)
            mu_t = mu_t_1 + mu_delta
            # (11)
            sigma_t = sigma_t_1 + Sigma_delta + np.cov(x_t_1, delta_t) + np.cov(delta_t, x_t_1)

            """ https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf 
                Page 54(66 of 217)  """
            I = np.eye(N)
            C = 0  # np.zeros(N, N)
            T_inv = (1 / sigma_c ** 2) * C.T * C
            # KIT: (3.46)
            S = T_inv * np.linalg.inv(I + sigma_t * T_inv)

            # KIT: (3.45)
            E_x_t = 1 - (1 / (np.linalg.det(I + sigma_t * T_inv)) ** 2) * np.exp(
                -0.5 * (mu_t - x_target).T * S * (mu_t - x_target))

            return E_x_t

        summe = 0
        N = dyn_model.N
        # (2)
        for t in range(1, N + 1):
            x = dyn_model.x
            summe = summe + E_x_t(x)
        J = summe

        return J

    def get_dJ(self, Theta):
        """
        Returns a function which can estimate the gradient of the expected return
        :param Theta: Policy param.s
        :return: function dJ
        """

        def dJ(Ext):
            """
            :param Ext: Expected returns
            :return: Gradient of expected returns w.r.t. policy param.s
            """
            # TODO: Torch grads
            # TODO: Maybe other deriv.s are also needed despite torch?
            # (26) Derivative of expected returns w.r.t. policy params
            dExt = grad(Ext, Theta)
            return dExt

        return dJ

    def approximate_p_delta_t(self, dyn_model, policy):
        """
        Approximates the predictive t-step distribution
        :param dyn_model:
        :param policy:
        :return:
        """
        # calculate   mu_delta
        # init
        n = dyn_model.N  # input (number of training data points)
        D = dyn_model.s_dim  # s_dim
        x_s = [ax[:-1] for ax in dyn_model.x]  # Training data

        # Generate test input
        mu_x = np.random.normal(size=dyn_model.s_dim+1) # TODO: What actions?? (Maybe get from policy based on the state)
        #mu_x[-1] = dyn_model.x[ax][-1]
        mu_x[-1] = policy.get_action(mu_x[:-1])

        # Predict mu and sigma for all test inputs
        # mu_schlange(t-1) is the mean of the "test" input distribution p(x[t-1],u[t-1])
        pred_mu, pred_Sigma = dyn_model.gp.predict([mu_x], return_std=True)
        pred_results = (pred_mu[0], pred_Sigma[0])

        q = np.zeros((n, D))
        y = np.mat(dyn_model.y)  # output
        v = np.zeros((n, D))
        length_scale = dyn_model.lambs
        Lambda = [np.diag(np.array([length_scale[a]] * D)) for a in range(D)]
        Lambda_inv = [np.linalg.inv(Lamb) for Lamb in Lambda]
        # alpha = np.ones(1, D)    ### nur fuer test, noch nicht bestimmt
        # TODO: Maybe estimate alphas for each dim. in dyn_model
        alpha = np.array([dyn_model.alpha] * D)  # alpha is (1, D) matrix ?

        # calculate q_ai
        for i in range(n):
            # (16)
            v[i] = x_s[i] - pred_results[0]  # x_schlange and mu_schlange in paper, x_schlange is training input,
            # mu_schlange is the mean of the "test" input distribution p(x[t-1],u[t-1])
            # TODO: Maybe return diag. matrix Sigma in dyn_model
            Sigma_t_1 = np.diag(np.array([pred_results[1]] * D))

            for a in range(D):
                # (15)
                fract = (alpha[a] ** 2) / np.sqrt(np.linalg.det(Sigma_t_1 * Lambda_inv[a] + np.eye(D)))
                vi = v[i].reshape(-1, 1)
                expo = np.exp(
                    (-1 / 2) * np.dot(np.dot(vi.T, np.linalg.inv(Sigma_t_1 + Lambda[a])),
                                      vi))  # Sigma[t-1] is variances at time t-1 from GP
                q[i][a] = fract * expo

        # calculate K
        # TODO: Maybe fix bugs in K computation
        # Lambda = np.diag(length_scale)
        K, K_inv = [], []  # K for all input und dimension, each term is also a matrix for all input and output
        for a in range(D):
            K_dim = np.ones((n, n))  # K_dim tmp to save the result of every dimension
            for i in range(n):
                for j in range(i+1, n):
                    # (6)
                    curr_x = (x_s[i] - x_s[j]).reshape(-1, 1)
                    # x_s is x_schlange in paper,training input
                    #kern = (alpha[a] ** 2) * np.exp(-0.5 * ((x_s[i][a] - x_s[j][a]) ** 2) / length_scale[a])
                    #kern = (alpha[a] ** 2) * np.exp(-0.5 * (np.sum(x_s[i] - x_s[j]) ** 2) / length_scale[a])
                    kern = 1e-10 * np.exp(
                        -0.5 * (np.dot(np.dot(curr_x.T, Lambda_inv[a]), curr_x)))
                    K_dim[i][j] = kern
                    K_dim[j][i] = kern
            K.append(K_dim)
            K_inv.append(np.linalg.inv(K_dim))

        # calculate   beta, under (14)
        beta = np.zeros((n, D))
        for a in range(D):
            beta[:, a] = (K_inv[a] * y[:, a]).reshape(-1, ) # TODO: Fix (Values are too big)

        # calculate mu_delta (14)
        mu_delta = np.zeros(D)
        for a in range(D):
            mu_delta[a] = np.dot(beta[:, a].reshape(-1, 1).T, q[:, a].reshape(-1, 1))

        # calculate Sigma_delta
        Sigma_delta = np.zeros((D, D))
        for a in range(D):
            for b in range(D):
                # Compute Q
                # TODO: How to use all test data?? (Currently using only first one)
                Q = self.compute_Q(alpha, Lambda_inv, a, b, x_s, v, pred_results[0], np.diag(np.array([pred_results[1]] * D)))

                beta_a = beta[:, a].reshape(-1,1)
                beta_b = beta[:, b].reshape(-1,1)

                if a != b:
                    # (20)
                    E_delta = np.dot(np.dot(beta_a.T, Q), beta_b)
                    # (18) = (20)- mu_delta_a*mu_delta_b
                    Sigma_delta[a][b] = E_delta - mu_delta[a] * mu_delta[b]
                else:
                    # (23)
                    E_var = alpha[a] ** 2 - np.trace(K_inv[a] * Q)
                    # (20)
                    E_delta_sq = np.dot(np.dot(beta_a.T, Q), beta_a)
                    # (17)=(23)+(20)- mu_delta_a**2
                    Sigma_delta[a][a] = E_var + E_delta_sq - mu_delta[a] ** 2

        return mu_delta, Sigma_delta

    def compute_Q(self, alpha, Lambda_inv, a, b, x_s, v, mu_t, Sigma_t):
        """
        Computes Q from eq.(22)
        :param alpha: Alpha GP hyperparam
        :param Lambda_inv: Contains inverted diagonal matrices for each length scale of the GP
        :param a: Index
        :param b: Index
        :param x_s: Training inputs
        :param v: v from eq.(16)
        :param mu_t: Predicted mean for t-1
        :param Sigma_t: Predicted Sigma for t-1
        :return: n x n matrix Q
        """
        n = len(x_s)
        D = alpha.shape[0]

        # calculate Q
        Q = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                curr_xi = (x_s[i] - mu_t).reshape(-1, 1)
                # x_s is x_schlange in paper,training input
                kern_a = (alpha[a] ** 2) * np.exp(-0.5 * np.dot(np.dot(curr_xi.T, Lambda_inv[a]), curr_xi))
                curr_xj = (x_s[j] - mu_t).reshape(-1, 1)
                kern_b = (alpha[b] ** 2) * np.exp(-0.5 * np.dot(np.dot(curr_xj.T, Lambda_inv[b]), curr_xj))
                # TODO： k_a and k_b are both scalar but not matrix, see (6)

                # Compute R
                R = Sigma_t * (Lambda_inv[a] + Lambda_inv[b]) + np.eye(D)
                z_ij = np.dot(Lambda_inv[a], v[i].reshape(-1, 1)) + np.dot(Lambda_inv[b], v[j].reshape(-1, 1))
                # (22), Q_ij should be a scalar, Q is a n*n matrix.
                frac = kern_a * kern_b / np.sqrt(np.linalg.det(R))
                expo = np.exp(0.5 * np.dot(np.dot(np.dot(z_ij.T, np.linalg.inv(R)), Sigma_t), z_ij))
                Q[i][j] = frac*expo
        return Q
