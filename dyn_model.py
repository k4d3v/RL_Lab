import numpy as np
import torch
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
        self.beta = beta

        self.mean = []
        self.cov_f = self.squared_expo_kernel
        self.setup_sigma()

    def squared_expo_kernel(self, x, y):
        """
        An exponential squared kernel function
        :param x: Training input 1
        :param y: Training input 2
        :return: The kernel function
        """
        # TODO: Learn length-scales delta_d for each dimension
        if self.lambs is None:
            lambs = np.array([self.s_dim+1]*(self.s_dim+1))
            return np.exp(-1 / 2.0 * np.sum(np.power((x - y) / lambs, 2)))
        else:
            return np.exp(-1 / 2.0 * np.sum(np.power((x - y) / self.lambs, 2)))

    def calculate_sigma(self, x, cov_f, beta=0.0):
        """
        Computes the covariance matrix
        :param x: Training inputs
        :param cov_f: GP kernel function
        :param beta: Hyperparam. for the covariance matrix
        :return: Covariance sigma
        """
        # Length of training data
        N = len(x)
        sigma = np.ones((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                # Compute kernels
                cov = cov_f(x[i], x[j])
                sigma[i][j] = cov
                sigma[j][i] = cov

        # Add beta hyperparam.
        sigma = sigma + beta * np.eye(N)
        return sigma

    def setup_sigma(self):
        """
        Sets up the covariance matrix
        """
        self.sigma = self.calculate_sigma(self.x, self.cov_f, self.beta)

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
        """
        # Compute derivatives of marginal ll
        sigma = np.mat(self.sigma.detach().numpy())
        K_inv = sigma.I
        alpha = K_inv*np.mat(self.y)
        nabla_logll = np.zeros(sigma.shape)
        for r in range(sigma.shape[0]):
            for c in range(sigma.shape[1]):
                gradi = grad(self.sigma[r][c], lambs, allow_unused=True)
                nabla_logll[r][c] = (1/2)*np.trace((alpha*alpha.T-K_inv)*gradi)

        # Optimize marginal ll
        """
        return lambs
