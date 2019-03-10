""" An implementation of the PILCO algorithm as shown in
http://www.icml-2011.org/papers/323_icmlpaper.pdf"""

import gym
import quanser_robots
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt

from gp_policy import GPPolicy as Policy
from dyn_model import DynModel


class PILCO:
    def __init__(self, env_name, J=2, N=10, T_init=50):
        """
        :param env_name: Name of the environment
        :param J: Number of rollouts
        :param N: Number of iterations
        :param T_init: Initial time horizon
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.env.seed(1)
        self.J = J
        self.N = N
        self.T = T_init

    def train(self):
        """
        Perform PILCO algorithm on initial random RBF policy
        :return: Optimal policy
        """
        start = timer()

        # Dimension of states
        s_dim = self.env.observation_space.shape[0]

        # Initial J random rollouts
        data = []
        # Sample controller params
        policy = Policy(self.env)
        for j in range(self.J):
            # Apply random control signals and record data
            data.append(policy.rollout(True))
            # Delete redundant states across trajectories
            data = self.regularize(data)

        old_policy = policy

        # Controlled learning (N iterations)
        for n in range(self.N):
            print("Round ", n)

            # Learn GP dynamics model using all data (Sec. 2.1)
            dyn_model = DynModel(s_dim, data)
            print("Average GP error: ", dyn_model.training_error_gp())
            # Plot learnt model
            dyn_model.plot()

            i = 0
            while True:
                print("Policy search iteration ", i)

                # Loop over number of params
                for p in range(policy.n_params):
                    # Approx. inference for policy evaluation (Sec. 2.2)
                    # Get J^pi(policy) (10-12), (24)
                    # J = self.get_J(dyn_model, all_params, p)
                    J = self.get_J(dyn_model, None, p)

                    # Update policy (CG or L-BFGS)
                    policy.update(J, -1)

                # Convergence check
                if policy.check_convergence(old_policy):
                    # Plot policy if converged
                    # policy.plot_rbf_net()
                    # Increase time horizon
                    self.T = int(self.T * 1.25)
                    break

                old_policy = policy
                i += 1

            # Apply new optimal policy to system (One episode) and record
            data.append(policy.rollout())
            data = self.regularize(data)

        print("Training done, ", timer() - start)
        return policy

    def get_J(self, dyn_model, all_params, p):
        """
        Returns a function which constructs a gaussian approximation for every p(x_t) based on subsequent one-step predictions and computes the expected values
        :param dyn_model: Trained dynamics model
        :param all_params: Contains parameters which are to be kept constant during optimization
        :param p: Starting index of the parameters for optimization
        :return: Function for estimating J (Expected values)
        """

        def J(param_array):
            """
            A function which computes the expected return for a policy
            :param param_array: Current policy params
            :return: expected return
            """
            astart = timer()

            # print("Current x:", param_array)

            def compute_E_x_t(mu_t, sigma_t):
                """
                Computes the expected return for specific mu and sigma for a time step
                :param mu_t: Mean at time step t
                :param sigma_t: Sigma at time step t
                :return: E_x_t : expected return at t
                """
                # start = timer()

                # defined according to the model
                x_target = np.array([0, 0, -1])  # target state
                sigma_c = 0.25  # control the width of cost function, top right of page 4
                # in KIT's paper,sigma_c is represented by a,in the example on page 64, a=  0.25
                l_p = 0.6413  # pendulum length, l_p = 0.6413 m, see User Manual

                # https://pdfs.semanticscholar.org/c9f2/1b84149991f4d547b3f0f625f710750ad8d9.pdf Page 54(66 of 217)
                C = np.mat(np.array([[1, l_p, 0], [0, 0, l_p]]))
                T_inv = (sigma_c ** -2) * C.T * C  # TODO: Inverse is too big!
                # T_inv = np.eye(3)

                # Use only first 3 dims
                mu_t = mu_t[:-2]
                sigma_t = sigma_t[:-2].T
                sigma_t = sigma_t[:-2].T

                # KIT: (3.46)
                IST = np.eye(3) + np.dot(sigma_t, T_inv)
                S = np.dot(T_inv, np.linalg.inv(IST))

                # KIT: (3.45)
                fact = 1 / np.sqrt(np.linalg.det(IST))
                # print("Cost factor (Should be close to 1, if sigma is small): ", fact)
                expo = np.exp(-0.5 * np.dot(np.dot((mu_t - x_target).T, S), (mu_t - x_target)))
                E_x_t = 1 - fact * expo

                # E_x_t should be 0, if target is reached!
                # For debugging
                E_x_t_target = 1 - fact * np.exp(
                    -0.5 * np.dot(np.dot((x_target - x_target).T, S), (x_target - x_target)))

                # print("Ext done", timer() - start)
                return E_x_t

            Ext_sum = 0

            # Reconstruct policy
            apolicy = Policy(self.env)
            if all_params is None:
                apolicy.assign_Theta(param_array)
            else:
                if p == 0:
                    all_params[:apolicy.s_dim * apolicy.n_basis] = param_array
                elif p == 1:
                    all_params[apolicy.s_dim * apolicy.n_basis:] = param_array
                apolicy.assign_Theta(all_params)

            # Store some vars for convenience
            self.prepare(apolicy)
            print("Inputs range: ", (np.min(apolicy.x), np.max(apolicy.x)))
            print("Targets range: ", (np.min(apolicy.y), np.max(apolicy.y)))

            # Generate initial test input
            # First state is known, predictive mean of action can be computed and variance is zero (p.44 Deisenroth)
            x0 = dyn_model.x[0][:-1]
            # Get action from policy based on the state
            x0 = np.concatenate((x0, apolicy.get_action(x0)))

            x_t_1 = x0
            # mu_t_1 = x0[:-1] + pred_mu[0]
            # sigma_t_1 = np.diag([pred_Sigma[0]] * apolicy.s_dim)
            mu_t_1 = x0[:-1]
            sigma_t_1 = np.diag([0] * apolicy.s_dim)

            # Compute mu_t for t from 1 to T
            # (10)-(12)
            print("Number of time steps: ", self.T)
            traj = [x0]
            for t in range(self.T):
                # print("Time step ", t)

                # (2)
                Ext_sum += compute_E_x_t(mu_t_1, sigma_t_1)

                # under 2.1
                # (10)
                pred_mu, pred_Sigma = dyn_model.predict([x_t_1])  # Predict mean and var for next state
                mu_t = mu_t_1 + pred_mu[0]

                # (11)
                # TODO: Fix (See Deisenroth)
                # Compute eigenvalues (for debugging)
                sigma_t = np.diag(pred_Sigma[0])
                ew, _ = np.linalg.eig(sigma_t)

                """
                # Failed try at approximating change in state
                # See Deisenroth p. 45f. and compute_succ in pilco_lin
                # Next state is gaussian distributed,
                # so the predictive mean and covariance of the action have to be computed (Deisenroth p.45)
                mu_u, Sigma_u, crosscov = self.approximate_p_delta_t(dyn_model, apolicy, x_t_1)
                mu_squashed_u = np.exp(-Sigma_u/2)*apolicy.a_max*np.sin(mu_u)

                # Compute joint distribution of x_t_1 and the unsquashed action distribution
                mu_joint = np.concatenate((mu_t, mu_u))
                Sigma_joint = np.block([[sigma_t, crosscov], [crosscov.T, Sigma_u]])
                
                # TODO: Rest of procedure
                mu_t = mu_t_1 + mu_delta
                sigma_t = sigma_t_1 + Sigma_delta + cov+cov.T
                """

                x_t = np.random.multivariate_normal(mu_t, sigma_t)  # Sample state x from predicted distribution
                # Get action from policy based on the state
                x_t = np.concatenate((x_t, apolicy.get_action(x_t)))

                # Update x, mu and sigma
                traj.append(x_t)
                x_t_1 = x_t
                mu_t_1 = mu_t
                sigma_t_1 = sigma_t

            # (2)
            Ext_sum += compute_E_x_t(mu_t_1, sigma_t_1)

            print("Expected costs: ", Ext_sum.item())
            print("J done, ", timer() - astart)

            # Plot trajectory
            # self.plot_traj(traj)

            self.Ext_sum = Ext_sum
            return Ext_sum

        # Return the function for computing the expected returns
        return J

    def approximate_p_delta_t(self, dyn_model, policy, x):
        """
        Approximates the predictive distribution for a Gaussian-sampled x
        :param dyn_model: GP dynamics model
        :param policy: Current policy
        :param x: Test input value
        :return: mu and sigma delta and cov of the predictive distribution
        """
        #start = timer()

        # init
        n = policy.n_basis # input (number of training data points)
        iD = policy.s_dim
        D = policy.a_dim  # s_dim

        # Predict mu and sigma for test input
        # mu_schlange(t-1) is the mean of the "test" input distribution p(x[t-1],u[t-1])
        pred_mu, pred_Sigma = dyn_model.predict([x])
        pred_results = (x[:-1] + pred_mu[0], pred_Sigma[0])
        Sigma_t_1 = np.diag(pred_results[1])

        # Plot prediction
        # dyn_model.plot(x, pred_mu, pred_Sigma)

        q = np.zeros((n, D))
        y = policy.y.reshape(-1, 1) # Training outputs
        v = np.zeros((n, iD))

        # calculate q_ai
        for i in range(n):
            # (16)
            v[i] = self.x_s[i] - pred_results[0]  # x_schlange and mu_schlange in paper, x_schlange is training input,
            # mu_schlange is the mean of the "test" input distribution p(x[t-1],u[t-1])

            for a in range(D):
                # (15)
                fract = (self.alpha ** 2) / np.sqrt(np.linalg.det(np.dot(Sigma_t_1, self.Lambda_inv[a]) + np.eye(D)))
                vi = v[i].reshape(-1, 1)
                # Sigma[t-1] is variances at time t-1 from GP
                expo = np.exp((-1 / 2) * np.dot(np.dot(vi.T, np.linalg.inv(Sigma_t_1 + self.Lambda[a])), vi))
                q[i][a] = fract * expo

        # calculate beta, under (14)
        # calculate mu_delta (14)
        beta = np.zeros((n, D))
        mu_delta = np.zeros(D)
        for a in range(D):
            beta[:, a] = np.dot(self.K_inv[a], y).reshape(-1, )
            mu_delta[a] = np.inner(beta[:, a], q[:, a]).reshape(-1, )

        # calculate Sigma_delta (Is symmetric!)
        Sigma_delta = np.zeros((D, D))
        for a in range(D):
            beta_a = beta[:, a].reshape(-1, 1)
            for b in range(a, D):
                # Compute Q
                Q = self.compute_Q(a, b, v, pred_results[0], Sigma_t_1)

                beta_b = beta[:, b].reshape(-1, 1)

                # (20)
                E_delta = np.dot(np.dot(beta_a.T, Q), beta_b)
                mu_prod = mu_delta[a] * mu_delta[b]
                if a != b:
                    # (18) = (20)- mu_delta_a*mu_delta_b
                    entry = E_delta - mu_prod
                    Sigma_delta[a][b] = entry
                    Sigma_delta[b][a] = entry
                else:
                    # (23)
                    E_var = self.alpha ** 2 - np.trace(np.dot(self.K_inv[a], Q)) + self.noise
                    # (17)=(23)+(20)- mu_delta_a**2
                    Sigma_delta[a][b] = E_var + E_delta - mu_prod if D>1 else E_delta-mu_prod

        # Compute eigenvals for debugging
        ew, _ = np.linalg.eig(Sigma_delta)

        # Compute covariance matrix (See (12)/ 2.70 Deisenroth)
        cov = np.zeros((iD, D))
        for a in range(D):
            sig_inver = np.dot(Sigma_t_1, np.linalg.inv(Sigma_t_1 + self.Lambda[a]))
            acol = 0
            for i in range(n):
                bq = beta[:, a][i] * q[:, a][i]
                x_mu = self.x_s[i] - pred_results[0]
                prod = np.dot((bq * sig_inver), x_mu)
                acol += prod
            cov[:, a] = acol

        # For debugging
        # Sigma_delta = Sigma_t_1
        # cov = np.zeros((iD,D))

        #print("Done approximating mu and sigma delta, ", timer() - start)
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
        #start = timer()

        n = len(self.x_s)
        D = len(self.x_s[0])

        R = np.dot(Sigma_t, (self.Lambda_inv[a] + self.Lambda_inv[b])) + np.eye(D)

        R_inv = np.linalg.inv(R)
        detsqrt = np.sqrt(np.linalg.det(R))

        # calculate Q
        Q = np.zeros((n, n))
        # Q_old = np.zeros((n, n))
        for i in range(n):
            ksi_i = (self.x_s[i] - mu_t).reshape(-1, 1)
            for j in range(i, n):
                # Deisenroth implementation (eq. 2.53)
                ksi_j = (self.x_s[j] - mu_t).reshape(-1, 1)
                z_ij = np.dot(self.Lambda_inv[a], ksi_i) + np.dot(self.Lambda_inv[b], ksi_j)

                fst = 2 * (np.log(self.alpha) + np.log(self.alpha))

                snd_1 = np.dot(np.dot(ksi_i.T, self.Lambda_inv[a]), ksi_i).item()
                snd_2 = np.dot(np.dot(ksi_j.T, self.Lambda_inv[b]), ksi_j).item()
                snd_3 = np.dot(np.dot(np.dot(z_ij.T, R_inv), Sigma_t), z_ij).item()

                snd = 0.5 * (snd_1 + snd_2 - snd_3)
                entry = np.exp(fst - snd) / detsqrt
                Q[i][j] = entry
                Q[j][i] = entry

        #print("Done estimating Q, ", timer() - start)
        return Q

    def prepare(self, policy):
        """
        Define some important vars
        :param policy: GP policy
        """
        self.D = policy.s_dim
        length_scale = policy.lambs
        self.Lambda = [np.diag(np.array([length_scale[a]] * self.D)) for a in range(self.D)]
        self.Lambda_inv = [np.linalg.inv(Lamb) for Lamb in self.Lambda]
        self.alpha = policy.alpha
        self.x_s = policy.x  # Training data
        self.noise = policy.noise
        # Compute Gram matrix and its inverse for each dimension
        self.compute_K()

    def compute_K(self):
        """
        Precomputes the Gram matrix of the policy and its inverse for each dimension
        """
        start = timer()

        D = self.D
        n = len(self.x_s)

        # Calculate K
        K, K_inv = [], []  # K for all input und dimension, each term is also a matrix for all input and output
        for a in range(D):
            K_dim = np.full((n, n), self.alpha ** 2)  # K_dim tmp to save the result of every dimension
            for i in range(n):
                for j in range(i + 1, n):
                    # (6)
                    curr_x = (self.x_s[i] - self.x_s[j]).reshape(-1, 1)
                    # self.x_s is x_schlange in paper,training input
                    kern = (self.alpha ** 2) * np.exp(-0.5 * (np.dot(np.dot(curr_x.T, self.Lambda_inv[a]), curr_x)))
                    K_dim[i][j] = kern
                    K_dim[j][i] = kern
            K.append(K_dim)
            K_inv.append(np.linalg.inv(K_dim + self.noise * np.eye(n)))

        self.K = K
        self.K_inv = K_inv
        print("Done precomputing K and K^-1, ", timer() - start)

    def regularize(self, data):
        """
        Removes redundant entries from data based on distances
        :param data: Samples from rollouts of policy
        :return: Sparse data
        """
        # Remove redundant entries across the trajectories
        for atraj in data[:-1]:
            states = [point[0] for point in atraj]
            for s in states:
                # Start at second trajectory point
                i = 1
                states_new = [point[0] for point in data[-1]]
                while i < len(states_new):
                    # Delete redundant state from last traj
                    if np.all(np.abs(s - states_new[i]) < 5e-2):
                        # Store action
                        data[-1][i - 1][1] += data[-1][i][1]

                        print("Deleted redundant state.")
                        del data[-1][i]
                        del states_new[i]
                    # Increment i if no redundancy was found
                    else:
                        i += 1
        print("New trajectory length:", len(data[-1]))
        return data

    def plot_traj(self, traj):
        """
        Plots a trajectory in s_dim+a_dim subplots
        :param traj: Trajectory containing (s,a) pairs
        """
        x = range(len(traj))
        for d in range(self.D + 1):
            plt.subplot(self.D + 1, 1, d + 1)
            y_d = np.array([ay[d] for ay in traj])
            plt.plot(x, y_d)
            plt.xlabel("Time Step")
            plt.ylabel("Pred. s " + str(d)) if d < self.D else plt.ylabel("Pred. a")
        plt.suptitle("Predicted Trajectories for each Dimension")
        plt.show()
