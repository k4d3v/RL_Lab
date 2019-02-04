import numpy as np
import torch
from scipy.optimize import minimize
from torch.autograd import Variable, grad
from torch.distributions import Normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from matplotlib import pyplot as plt


class DynModel:
    """
    Gaussian Process with a squared exponential kernel for learning the system dynamics
    """

    def __init__(self, s_dim, data, lambs=None):
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
        self.beta = 0.0001

        self.cov_f = self.squared_expo_kernel
        #self.setup_sigma()
        self.fit_gp()

    def squared_expo_kernel(self, x, y, lambs=None):
        """
        An exponential squared kernel function
        :param x: Training input 1
        :param y: Training input 2
        :return: The kernel function
        :param lambs: Custom length-scales (l^2)
        """
        if lambs is None:
            lambs = np.array([self.s_dim + 1] * (self.s_dim + 1))
        return (self.alpha ** 2) * np.exp(-1 / 2.0 * np.sum(np.power((x - y), 2) / lambs))

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
        self.sigma = self.calculate_sigma(self.x, self.cov_f, self.lambs)

    def predict(self, x):
        """
        Predicts output values
        :param x: Input
        :return: Mu and Sigma of prediction
        """
        # Kernel of input
        k_xx = 1 + self.beta * self.cov_f(x, x, self.lambs)
        # Kernel of input and training x
        k_xX = np.zeros((self.N, 1))
        for i in range(self.N):
            k_xX[i] = self.cov_f(self.x[i], x, self.lambs)

        # See slide 43 LR ML2 lecture
        sigmaI = np.mat(self.sigma).I
        m_expt = (k_xX.T * sigmaI) * np.mat(self.y)
        # sigma_expt = k_xx - (k_xX.T * np.mat(self.sigma).I) * k_xX
        sigma_expt = k_xx + self.beta - (k_xX.T * sigmaI) * k_xX
        return np.array(m_expt).reshape(-1, ), np.array(sigma_expt).reshape(-1, )

    def prepare_data(self, data):
        """
        Extracts (s_t-1, a_t-1) as x and s_t as y from given training data
        :param data: Trajectories containing points (s, a, r)
        :return: x, y training data
        """
        x, y = [], []
        for traj in data:
            for p in range(len(traj) - 1):
                # Join state and action
                x.append(np.concatenate((traj[p][0], traj[p][1])))
                y.append(traj[p + 1][0] - traj[p][0])
        return x, y

    def estimate_hyperparams(self):
        """
        Estimates the optimal length-scales for the kernel by maximizing the marginal log-likelihood
        :return: optimal length-scales
        """
        sig_n = 0.1 ** 2  # TODO: Is it really needed?
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
                data_fit = (-1 / 2) * y[:, d].T * KI * y[:, d]
                det = (1 / 2) * np.log(np.linalg.det(K))
                norm = -(n / 2) * np.log(2 * np.pi)
                log_prob.append(data_fit - det - norm)
            return np.mean(log_prob)

        # Optimize marginal ll
        init = [1] * (self.s_dim + 1)
        bounds = [(1e-9, None)] * (self.s_dim + 1)
        # TODO: Does minimization work right though?
        optimal_lambs = minimize(mll, init, method='L-BFGS-B', bounds=bounds, options={'disp': True}).x
        return optimal_lambs

    def training_error(self):
        """
        Estimates the average error on the training data
        :return: Average error
        """
        err = []
        for din, dout in zip(self.x, self.y):
            m, sig = self.predict(din)
            err.append(np.abs(dout - m))
        return np.sum(err) / len(self.x)

    def training_error_gp(self):
        """
        Estimates the average error on the training data
        :return: Average error
        """
        err = []
        m = self.gp.predict(self.x)
        for i, dout in zip(range(len(self.x)), self.y):
            err.append(np.abs(dout - m[i]))
        return np.sum(err) / len(self.x)

    def fit_gp(self):
        """
        Fits a sklearn GP based on the training data
        """
        kern = ConstantKernel(self.alpha**2, constant_value_bounds=(1, 10)) \
               * RBF(length_scale=[1] * (self.s_dim + 1), length_scale_bounds=np.array([1e-5, 1])) \
               + WhiteKernel(noise_level=1e-5)
        #gp = GaussianProcessRegressor(kernel=kern, optimizer=None)
        gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=10)
        gp.fit(self.x, self.y)
        # self.lambs = gp.kernel_.get_params()["length_scale"]
        opti_params = gp.kernel_.get_params()
        self.alpha = np.sqrt(opti_params["k1"].k1.constant_value)
        self.lambs = opti_params["k1"].k2.length_scale
        self.noise = opti_params["k2"].noise_level
        self.gp = gp
        print("GPML kernel: %s" % gp.kernel_)
        print(gp.kernel_.get_params())
        print("Done fitting GP")

    def plot(self, x_test=None, y_pred=None, sigma=None):
        """
        Plots ground truth and predictions
        :param x_test: A gaussian distributed test input
        :param y_pred: Predicted mean
        :param sigma: Predicted std
        """
        # Predict on training inputs
        if y_pred is None:
            mu, sig = self.gp.predict(self.x, return_std=True)
            plt.suptitle("Prediction on Training Data")
        else:
            plt.suptitle("Training vs Test")

        # Plot the function, the prediction and the 95% confidence interval based on
        # the MSE
        for d in range(self.s_dim):
            plt.subplot(self.s_dim + 1, 1, d + 1)

            x_d = np.array([ax[d] for ax in self.x])
            y_d = np.array([ay[d] for ay in self.y])
            plt.plot(x_d, y_d, 'r.', markersize=10, label=u'Observations')
            if not y_pred is None:
                x_test_d = np.array([x_test[d]])
                y_pred_d = np.array([y_pred[0][d]])
                plt.plot(x_d, y_d, 'b-', label=u'Training Data')
                plt.errorbar(x_test_d.ravel(), y_pred_d, sigma, fmt='g.', markersize=10, label=u'Prediction')
            else:
                mu_d = np.array([ay[d] for ay in mu])
                plt.plot(x_d, mu_d, 'b-', label=u'Prediction')
                plt.fill(np.concatenate([x_d, x_d[::-1]]),
                         np.concatenate([mu_d - 1.9600 * sig, (mu_d + 1.9600 * sig)[::-1]]),
                         alpha=.5, fc='b', ec='None', label='95% confidence interval')

            plt.xlabel("In " + str(d))
            plt.ylabel("Out " + str(d))
            plt.legend(loc='lower left')
        plt.show()
