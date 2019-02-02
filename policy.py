import numpy as np
from timeit import default_timer as timer
import torch
from scipy.optimize import minimize
from torch.autograd import Variable


class Policy():
    """
    Represents a RBF policy
    """
    def __init__(self, env, n_basis=50):
        """
        Nonlinear RBF network, used as a state-feedback controller
        :param env: Environment
        :param n_basis: Number of basis functions
        """
        self.env = env
        self.s_dim = self.env.observation_space.shape[0]
        self.n_basis = n_basis
        # Init. random control param.s
        W = Variable(torch.rand(self.n_basis), requires_grad=True)
        Lamb = Variable(torch.Tensor(np.eye(self.s_dim)), requires_grad=True)
        Mu = Variable(torch.rand(self.n_basis, self.s_dim), requires_grad=True)
        self.Theta = {"W": W, "Lamb": Lamb, "Mu": Mu}

    def get_action(self, x):
        """
        Returns a single control based on observation x
        :param x: Observation
        :return: Control
        """
        sum = 0
        for i in range(self.n_basis):
            sum += self.Theta["W"].detach().numpy()[i]*self.calc_feature(x, i)
        return np.array([sum])

    def calc_feature(self, x, i):
        """
        Calculates a basis function feature
        :param x: Observation
        :param i: Number of current basis function
        :return: phi_i(x)
        """
        phi_x = np.exp(
            -0.5*np.dot(np.dot(np.transpose(x-self.Theta["Mu"].detach().numpy()[i]).T,
                               np.linalg.inv(self.Theta["Lamb"].detach().numpy())),
                        (x-self.Theta["Mu"].detach().numpy()[i])))
        return phi_x

    def rollout(self):
        """
        Samples a traj from performing actions based on the current policy
        :return: Sampled trajs
        """
        start = timer()

        old_observation = [np.inf]*self.s_dim

        # Reset the environment
        observation = self.env.reset()
        episode_reward = 0.0
        done = False
        traj = []

        while not done:
            # Show environment
            #self.env.render()
            point = []

            action = self.get_action(np.asarray(observation))

            point.append(observation)  # Save state to tuple
            point.append(action)  # Save action to tuple
            observation, reward, done, _ = self.env.step(action)  # Take action
            point.append(reward)  # Save reward to tuple

            episode_reward += reward

            # Append point if it is far enough from the previous one
            if not np.all(np.abs(observation - old_observation) < 5e-2):
                traj.append(point)  # Add Tuple to traj
                old_observation = observation
            else:
                print("Sampled redundant state.")

        print("Done rollout, ", timer() - start)
        print("Episode reward: ", episode_reward)
        return traj

    def update(self, J, dJ):
        """
        Optimizes the policy param.s w.r.t. the expected return
        :param J: Function for computing the expected return
        :param dJ: Function for computing the gradient of the expected return
        """
        init = self.param_array()
        bnds = ([(1e-5, 1)]*self.n_basis + [(1e-5, 0.2)]*self.s_dim + [(1e-5, 1)]*(self.n_basis*self.s_dim))
        #new_Theta = minimize(J, init, method='L-BFGS-B', jac=dJ, bounds=bnds options={'disp': True}).x
        new_Theta = minimize(J, init, method='L-BFGS-B', bounds=bnds, options={'disp': True}).x
        print("Optimization of policy params done.")
        self.assign_Theta(new_Theta)

    def check_convergence(self, old_Theta):
        """
        Checks if there is a significant difference between the old and new policy param.s
        :param old_Theta: Old params
        :return: True if convergence
        """
        return True

    def param_array(self):
        """
        Returns an array containing all the policy parameters
        :return: np array with the policy params
        """
        pl = list(self.Theta.values())
        pl = [p.detach().numpy() for p in pl]
        # Store only diagonal of Lambda
        pl[1] = np.diag(pl[1])
        params = [list(param.flatten()) for param in pl]
        return np.array(params[0]+params[1]+params[2])

    def assign_Theta(self, pa):
        """
        Reconstructs the parameters W, Mu, Lambda from the given array and sets them as policy parameters
        :param pa: parameters array
        """
        # Reconstruct policy params
        mui = self.n_basis+self.s_dim
        W = pa[:self.n_basis]
        lambdi = pa[self.n_basis:mui]
        Lamb = np.diag(lambdi)
        Mu = pa[mui:]
        Mu = Mu.reshape(self.n_basis, self.s_dim)

        # Assign policy params
        self.Theta["W"] = Variable(torch.Tensor(W), requires_grad=True)
        self.Theta["Lamb"] = Variable(torch.Tensor(Lamb), requires_grad=True)
        self.Theta["Mu"] = Variable(torch.Tensor(Mu), requires_grad=True)
