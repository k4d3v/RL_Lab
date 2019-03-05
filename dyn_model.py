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

    def __init__(self, s_dim, data):
        """
        :param s_dim: Dimension of states
        :param data: Training inputs+outputs
        :param lambs: Hyperparam. for kernels (length-scale)
        """
        super().__init__()
        self.s_dim = s_dim
        self.x, self.y = self.prepare_data(data)
        self.N = len(self.x)

        self.alpha = np.ones(self.s_dim)
        self.lambs = np.ones(self.s_dim)
        self.noise = np.full(self.s_dim, 1e-3)
        self.gp = []

        # Fit a GP for each dimension
        for d in range(self.s_dim):
            self.fit_gp(d)

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

    def training_error_gp(self):
        """
        Estimates the average error on the training data
        :return: Average error
        """
        err = []
        m, _ = self.predict(self.x)
        for i, dout in zip(range(len(self.x)), self.y):
            err.append(np.abs(dout - m[i]))
        return np.sum(err) / len(self.x)

    def fit_gp(self, d):
        """
        Fits a sklearn GP based on the training data
        :param d: Current dimension
        """
        kern = ConstantKernel(self.alpha[d]**2, constant_value_bounds=(1e-5, 9)) \
               * RBF(length_scale=self.lambs[d], length_scale_bounds=np.array([0.1, 2])) \
               + WhiteKernel(noise_level=self.noise[d])
        #gp = GaussianProcessRegressor(kernel=kern, optimizer=None)
        gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=10)
        gp.fit([[x[d]] for x in self.x], [[y[d]] for y in self.y])
        opti_params = gp.kernel_.get_params()
        self.alpha[d] = np.sqrt(opti_params["k1"].k1.constant_value)
        self.lambs[d] = opti_params["k1"].k2.length_scale
        self.noise[d] = opti_params["k2"].noise_level
        self.gp.append(gp)
        print("GPML kernel: %s" % gp.kernel_)
        #print(gp.kernel_.get_params())
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
            mu, sig = self.predict(self.x)
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
                plt.errorbar(x_test_d.ravel(), y_pred_d, sigma[d], fmt='g.', markersize=10, label=u'Prediction')
            else:
                mu_d = np.array([ay[d] for ay in mu])
                plt.plot(x_d, mu_d, 'b-', label=u'Prediction')
                plt.fill(np.concatenate([x_d, x_d[::-1]]),
                         np.concatenate([mu_d - 1.9600 * sig[:, d], (mu_d + 1.9600 * sig[:, d])[::-1]]),
                         alpha=.5, fc='b', ec='None', label='95% confidence interval')

            plt.xlabel("In " + str(d))
            plt.ylabel("Out " + str(d))
            plt.legend(loc='lower left')
        plt.show()

    def predict(self, x):
        """
        Predicts values for a given list of inputs
        :param x: List of input values
        :return: Predicted targets
        """
        mu_all, sig_all = np.zeros((len(x), self.s_dim)), np.zeros((len(x), self.s_dim))
        for d in range(self.s_dim):
            mu, sig = self.gp[d].predict([[ax[d]] for ax in x], return_std=True)
            mu_all[:, d] = mu.reshape(-1,)
            sig_all[:, d] = sig
        return mu_all, sig_all
