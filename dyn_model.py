import numpy as np
import torch
from scipy.optimize import minimize
from torch.autograd import Variable, grad
from torch.distributions import Normal


class DynModel:
    """
    Gaussian Process with a squared exponential kernel for learning the system dynamics
    """
    def __init__(self, s_dim, data, lambs=None, beta=0.0001):
        """
        :param s_dim: Dimension of states
        :param x: Training inputs
        :param y: Training outputs
        :param lambs: Hyperparam. for kernels (length-scale)
        :param beta: Hyperparam. for covariance matrix
        """
        super().__init__()
        self.s_dim = s_dim
        self.x, self.y = self.prepare_data(data)
        self.N = len(self.x)
        self.lambs = lambs
        self.alpha = 1
        self.beta = beta

        self.mean = []
        self.cov_f = self.squared_expo_kernel
        self.setup_sigma()

    def squared_expo_kernel(self, x, y, lambs=None):
        """
        An exponential squared kernel function
        :param x: Training input 1
        :param y: Training input 2
        :return: The kernel function
        """
        # TODO: Learn length-scales delta_d for each dimension
        if lambs is None:
            lambs = np.array([self.s_dim+1]*(self.s_dim+1))
        return self.alpha*np.exp(-1 / 2.0 * np.sum(np.power((x - y) / lambs, 2)))

    def calculate_sigma(self, x, cov_f, lambs=None):
        """
        Computes the covariance matrix
        :param x: Training inputs
        :param cov_f: GP kernel function
        :return: Covariance sigma
        """
        # Length of training data
        N = len(x)
        sigma = np.ones((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                # Compute kernels
                cov = cov_f(x[i], x[j], lambs)
                sigma[i][j] = cov
                sigma[j][i] = cov

        # Add beta hyperparam.
        sigma = sigma + self.beta * np.eye(N)
        return sigma

    def setup_sigma(self):
        """
        Sets up the covariance matrix
        """
        self.sigma = self.calculate_sigma(self.x, self.cov_f)

    def predict(self, x):
        """
        Predicts output values
        :param x: Input
        :return: Mu and Sigma of prediction
        """
        # Kernel of input
        k_xx = 1 + self.beta * self.cov_f(x, x)
        # Kernel of input and training x
        k_xX = np.zeros((self.N, 1))
        for i in range(self.N):
            k_xX[i] = self.cov_f(self.x[i], x)

        # See slide 43 LR ML2 lecture
        sigmaI = np.mat(self.sigma).I
        m_expt = (k_xX.T * sigmaI) * np.mat(self.y)
        # sigma_expt = k_xx - (k_xX.T * np.mat(self.sigma).I) * k_xX
        sigma_expt = k_xx + self.beta - (k_xX.T * sigmaI) * k_xX
        return m_expt, sigma_expt

    def prepare_data(self, data):
        """
        Extracts (s_t-1, a_t-1) as x and s_t as y from given training data
        :param data: Trajectories containing points (s, a, r)
        :return: x, y training data
        """
        x, y = [], []
        for traj in data:
            for p in range(len(traj)-1):
                # Join state and action
                x.append(np.concatenate((traj[p][0], traj[p][1])))
                y.append(traj[p+1][0])
        return x, y

    def estimate_hyperparams(self):
        """
        Estimates the optimal length-scales for the kernel by maximizing the marginal log-likelihood
        :return: optimal length-scales
        """
        lambs = [self.s_dim+1]*(self.s_dim+1)

        sig_n = 0.1**2
        y = np.mat(self.y)
        n = self.sigma.shape[0]

        # Function for computing the marginal likelihood
        def mll(lambs):
            # Compute Kernel matrix and inverse based on hyperparam lambda
            sigma = self.calculate_sigma(self.x, self.cov_f, lambs)
            K = np.mat(sigma) + sig_n * np.eye(sigma.shape[0])
            KI = K.I

            log_prob = []
            # Compute mll for each dimension
            for d in range(self.s_dim):
                data_fit = (-1/2)*y[:,d].T*KI*y[:,d]
                det = (1/2)*np.log(np.linalg.det(K))
                norm = -(n/2)*np.log(2*np.pi)
                log_prob.append(data_fit-det-norm)
            return np.mean(log_prob)

        """
        # Function for computing the derivatives of marginal ll
        def dmll(y):
            # TODO
            def gradi(K, lambs):
                return 0
            alpha = KI*y
            return (1/2)*np.trace((alpha*alpha.T-KI)*gradi(K, lambs))
        """

        # Optimize marginal ll
        init = [1]*(self.s_dim+1)
        optimal_lambs = minimize(mll, init, method='BFGS', options = {'disp': True}).x
        return optimal_lambs
