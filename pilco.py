""" An implementation of the PILCO algorithm as shown in
http://www.icml-2011.org/papers/323_icmlpaper.pdf"""
import gym
import quanser_robots
import numpy as np
from torch.autograd import grad
from timeit import default_timer as timer

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
        self.env = gym.make(env_name)
        self.J = J
        self.N = N

    def train(self):
        """
        Perform PILCO algorithm on initial random RBF policy
        :return: Optimal policy
        """
        start = timer()

        # Init. environment
        env = gym.make(self.env_name)
        self.env = env
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
                
                self.prepare(dyn_model)
                
                #mu_delta, Sigma_delta = self.approximate_p_delta_t(dyn_model, policy)  # TODO

                # Approx. inference for policy evaluation (Sec. 2.2)
                # Get J^pi(policy) (10-12), (24)
                J = self.get_J(dyn_model)  # TODO

                # Policy improvement based on the gradient (Sec. 2.3)
                # Get the gradient of J (26-30)
                # TODO: Torch gradient
                dJ = self.get_dJ(policy)

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

        print("Training done, ", timer()-start)
        return policy

    def get_J(self, dyn_model):
        """
        Returns a function which constructs a gaussian approximation for every p(x_t) based on subsequent one-step predictions and computes the expected values
        :param dyn_model: Trained dynamics model
        :return: Function for estimating J (Expected values)
        """
        def J(param_array):
            """
            :param param_array: Current policy params
            :return: expected return
            """
            astart = timer()
            def E_x_t(mu_t, sigma_t):
                """
                :param mu_t:
                :param sigma_t:
                :return: E_x_t : expected return at t
                """
                start = timer()

                D = dyn_model.s_dim

                # defined according to the model
                x_target = np.zeros(3)  # target state
                sigma_c = 0.25  # control the width of cost function, top right of page 4
                             # in KIT's paper,sigma_c is represented by a,in the example on page 64, a=  0.25
                l_p = 0.6413  # pendulum length, l_p = 0.6413 m, see User Manual

                """ https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf 
                    Page 54(66 of 217)  """
                I = np.eye(3)

                # TODO: what is C and T? Example see Page 64 (3.71)
                #C = np.zeros((D, D))  # C is related to l_p, the pendulum length
                #C = np.array([[1.0, l_p, 0.0], [0.0, 0.0, l_p]])
                #T_inv = (1 / sigma_c ** 2) * C.T * C
                """
                T_inv = np.zeros((D,D))
                T_inv[0][0] = 1
                T_inv[0][1] = l_p
                T_inv[1][0] = l_p
                T_inv[1][1] = l_p**2
                T_inv[2][2] = l_p**2
                T_inv *= sigma_c**(-2)
                """
                C = np.mat(np.array([[1, l_p, 0], [0, 0, l_p]]))
                T_inv = (sigma_c**-2)*C.T*C
                #T_inv = np.eye(D)

                # Use only first 3 dims
                mu_t = mu_t[:-2]
                sigma_t = sigma_t[:-2].T
                sigma_t = sigma_t[:-2].T

                # KIT: (3.46)
                S = T_inv * np.linalg.inv(I + sigma_t * T_inv)

                # KIT: (3.45)
                # TODO: fact is nan; Fix T!
                fact = 1 / np.sqrt(np.linalg.det(I + sigma_t * T_inv))
                expo = np.exp(
                    -0.5 * np.dot(np.dot((mu_t - x_target).T,  S), (mu_t - x_target)))
                E_x_t = 1 - fact * expo

                print("Ext done", timer()-start)
                return E_x_t

            Ext_sum = 0
            n = dyn_model.N

            # Reconstruct policy
            apolicy = Policy(self.env)
            apolicy.assign_Theta(param_array)

            # Generate initial test input
            x0 = np.random.normal(size=dyn_model.s_dim + 1)
            # TODO: What actions?? (Maybe get from policy based on the state)
            # mu_x[-1] = dyn_model.x[ax][-1]
            x0[-1] = apolicy.get_action(x0[:-1])

            pred_mu, pred_Sigma = dyn_model.gp.predict([x0], return_std=True)

            # Compute mu_t for t from 1 to n
            # (10)-(12)
            x_t_1 = x0
            mu_t_1 = pred_mu[0]
            sigma_t_1 = np.diag([pred_Sigma[0]]*apolicy.s_dim)
            for t in range(3):
                print("Time step ", t)

                mu_delta, Sigma_delta, cov = self.approximate_p_delta_t(dyn_model, x_t_1)
                # under 2.1
                # (10)
                mu_t = mu_t_1 + mu_delta
                # (11)
                # TODO: Sigma is not a diagonal matrix!
                x_t = [np.random.normal(mu_t_1[d], np.diag(sigma_t_1)[d]) for d in range(mu_t_1.shape[0])]
                x_t = np.array(x_t + list(apolicy.get_action(np.array(x_t))))
                #delta_t = x_t-x_t_1
                # TODO: Fix (See Deisenroth)
                sigma_t = sigma_t_1 + Sigma_delta + 2*cov

                # (2)
                Ext_sum += E_x_t(mu_t, sigma_t)

                # Update x, mu and sigma
                x_t_1 = x_t
                mu_t_1 = mu_t
                sigma_t_1 = sigma_t

            print("J done, ", timer()-astart)
            return Ext_sum

        return J

    def get_dJ(self, policy):
        """
        Returns a function which can estimate the gradient of the expected return
        :param policy:
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
            dExt = grad(Ext, policy.Theta)
            return dExt

        return dJ

    def approximate_p_delta_t(self, dyn_model, x):
        """
        Approximates the predictive t-step distribution
        :param dyn_model: GP dynamics model
        :param x: Test input value
        :return: mu and sigma delta and cov of the predictive distribution
        """
        start = timer()

        # calculate mu_delta
        # init
        n = dyn_model.N  # input (number of training data points)
        D = dyn_model.s_dim  # s_dim
        
        # Predict mu and sigma for all test inputs
        # mu_schlange(t-1) is the mean of the "test" input distribution p(x[t-1],u[t-1])
        pred_mu, pred_Sigma = dyn_model.gp.predict([x], return_std=True)
        pred_results = (pred_mu[0], pred_Sigma[0])

        q = np.zeros((n, D))
        y = np.mat(dyn_model.y)  # output
        v = np.zeros((n, D))

        # TODO: Maybe return diag. matrix Sigma in dyn_model
        Sigma_t_1 = np.diag(np.array([pred_results[1]] * D))

        # calculate q_ai
        for i in range(n):
            # (16)
            v[i] = self.x_s[i] - pred_results[0]  # x_schlange and mu_schlange in paper, x_schlange is training input,
            # mu_schlange is the mean of the "test" input distribution p(x[t-1],u[t-1])

            for a in range(D):
                # (15)
                fract = (self.alpha[a] ** 2) / np.sqrt(np.linalg.det(Sigma_t_1 * self.Lambda_inv[a] + np.eye(D)))
                vi = v[i].reshape(-1, 1)
                # Sigma[t-1] is variances at time t-1 from GP
                expo = np.exp(
                    (-1 / 2) * np.dot(np.dot(vi.T, np.linalg.inv(Sigma_t_1 + self.Lambda[a])), vi))
                q[i][a] = fract * expo

        # calculate K
        # TODO: Maybe fix bugs in K computation
        # self.Lambda = np.diag(length_scale)
        K, K_inv = [], []  # K for all input und dimension, each term is also a matrix for all input and output
        for a in range(D):
            K_dim = np.ones((n, n))  # K_dim tmp to save the result of every dimension
            for i in range(n):
                for j in range(i+1, n):
                    # (6)
                    curr_x = (self.x_s[i] - self.x_s[j]).reshape(-1, 1)
                    # self.x_s is x_schlange in paper,training input
                    #kern = (self.alpha[a] ** 2) * np.exp(-0.5 * ((self.x_s[i][a] - self.x_s[j][a]) ** 2) / length_scale[a])
                    #kern = (self.alpha[a] ** 2) * np.exp(-0.5 * (np.sum(self.x_s[i] - self.x_s[j]) ** 2) / length_scale[a])
                    kern = self.alpha[a]**2 * np.exp(
                        -0.5 * (np.dot(np.dot(curr_x.T, self.Lambda_inv[a]), curr_x)))
                    K_dim[i][j] = kern
                    K_dim[j][i] = kern
            K.append(K_dim)
            K_inv.append(np.linalg.inv(K_dim))

        # calculate beta, under (14)
        # calculate mu_delta (14)
        beta = np.zeros((n, D))
        mu_delta = np.zeros(D)
        for a in range(D):
            beta[:, a] = (K_inv[a] * y[:, a]).reshape(-1, ) # TODO: Fix (Values are too big)
            mu_delta[a] = np.dot(beta[:, a].reshape(-1, 1).T, q[:, a].reshape(-1, 1))
            
        # calculate Sigma_delta
        Sigma_delta = np.zeros((D, D))
        for a in range(D):
            for b in range(D):
                # Compute Q
                Q = self.compute_Q(a, b, v, pred_results[0], np.diag(np.array([pred_results[1]] * D)))

                beta_a = beta[:, a].reshape(-1,1)
                beta_b = beta[:, b].reshape(-1,1)

                if a != b:
                    # (20)
                    E_delta = np.dot(np.dot(beta_a.T, Q), beta_b)
                    # (18) = (20)- mu_delta_a*mu_delta_b
                    Sigma_delta[a][b] = E_delta - mu_delta[a] * mu_delta[b]
                else:
                    # (23)
                    E_var = self.alpha[a] ** 2 - np.trace(K_inv[a] * Q)
                    # (20)
                    E_delta_sq = np.dot(np.dot(beta_a.T, Q), beta_a)
                    # (17)=(23)+(20)- mu_delta_a**2
                    Sigma_delta[a][a] = E_var + E_delta_sq - mu_delta[a] ** 2

        # Compute covariance matrix (See (12)/ 2.70 Deisenroth)
        # TODO: Vectorize
        cov = np.zeros((D,D))
        for a in range(D):
            sig_inver = Sigma_t_1*np.linalg.inv(Sigma_t_1+self.Lambda[a])
            acol = 0
            for i in range(n):
                bq = beta[:, a][i]*q[:, a][i]
                x_mu = self.x_s[i]-pred_results[0]
                prod = np.dot((bq*sig_inver), x_mu)
                acol += prod
            cov[:, a] = acol

        print("Done approximating mu and sigma delta, ", timer()-start)
        return mu_delta, Sigma_delta, cov

    def compute_Q(self, a, b, v, mu_t, Sigma_t):
        """
        Computes Q from eq.(22)
        :param a: Index
        :param b: Index
        :param v: v from eq.(16)
        :param mu_t: Predicted mean for t-1
        :param Sigma_t: Predicted Sigma for t-1
        :return: n x n matrix Q
        """
        start = timer()

        n = len(self.x_s)
        D = self.alpha.shape[0]

        R = Sigma_t * (self.Lambda_inv[a] + self.Lambda_inv[b] + np.eye(D))
        print(Sigma_t[0][0])
        print(R[0][0])
        R_inv = np.linalg.inv(R)

        # calculate Q
        Q, Q_old = np.zeros((n, n)),  np.zeros((n, n))
        for i in range(n):
            ksi_i = self.x_s[i] - mu_t
            for j in range(n):
                # Deisenroth implementation (eq. 2.53)
                ksi_j = self.x_s[j] - mu_t
                z_ij = (np.dot(self.Lambda_inv[a], ksi_i) + np.dot(self.Lambda_inv[b], ksi_j)).reshape(-1,1)

                fst = 2*(np.log(self.alpha[a])+np.log(self.alpha[b]))

                snd_1 = np.dot(np.dot(ksi_i.T, self.Lambda_inv[a]), ksi_i)
                snd_2 = np.dot(np.dot(ksi_j.T, self.Lambda_inv[b]), ksi_j)
                snd_3 = np.dot(np.dot(np.dot(z_ij.T, R_inv), Sigma_t), z_ij)

                snd = 0.5 * (snd_1 + snd_2 - snd_3)
                Q[i][j] = np.exp(fst-snd)/np.sqrt(np.linalg.det(R))

                """
                curr_xi = (self.x_s[i] - mu_t).reshape(-1, 1)
                # self.x_s is x_schlange in paper,training input
                kern_a = 1e-10 * np.exp(-0.5 * np.dot(np.dot(curr_xi.T, self.Lambda_inv[a]), curr_xi))
                curr_xj = (self.x_s[j] - mu_t).reshape(-1, 1)
                kern_b = 1e-10 * np.exp(-0.5 * np.dot(np.dot(curr_xj.T, self.Lambda_inv[b]), curr_xj))

                # Compute R
                z_ij = np.dot(self.Lambda_inv[a], v[i].reshape(-1, 1)) + np.dot(self.Lambda_inv[b], v[j].reshape(-1, 1))
                # (22), Q_ij should be a scalar, Q is a n*n matrix.
                frac = kern_a * kern_b / np.sqrt(np.linalg.det(R))
                expo = np.exp(0.5 * np.dot(np.dot(np.dot(z_ij.T, R_inv), Sigma_t), z_ij))
                Q_old[i][j] = frac*expo
                """

        print("Done estimating Q, ", timer()-start)
        return Q

    def prepare(self, dyn_model):
        """
        Define some important vars
        :param dyn_model: GP dynamics model
        """
        self.D = dyn_model.s_dim
        length_scale = dyn_model.lambs
        self.Lambda = [np.diag(np.array([length_scale[a]] * self.D)) for a in range(self.D)]
        self.Lambda_inv = [np.linalg.inv(Lamb) for Lamb in self.Lambda]
        # alpha = np.ones(1, D)    ### nur fuer test, noch nicht bestimmt
        # TODO: Maybe estimate alphas for each dim. in dyn_model
        self.alpha = np.array([dyn_model.alpha] * self.D)  # alpha is (1, D) matrix ?
        self.x_s = [ax[:-1] for ax in dyn_model.x]  # Training data
