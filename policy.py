import numpy as np
from timeit import default_timer as timer
import torch
from scipy.optimize import minimize
from torch.autograd import Variable
from matplotlib import pyplot as plt


class Policy():
    """
    Represents a RBF policy
    """
    def __init__(self, env, plot_pol=False, n_basis=50):
        """
        Nonlinear RBF network, used as a state-feedback controller
        :param env: Environment
        :param n_basis: Number of basis functions
        """
        self.env = env
        self.s_dim = self.env.observation_space.shape[0]
        self.n_basis = n_basis
        self.n_params = 3

        # Init. random control param.s

        # Weights: Ranging from min to max action
        # TODO: How to ensure that sum of RBFs is between -25 and 25?
        self.w_min = self.env.action_space.low
        self.w_max = self.env.action_space.high
        ws = np.linspace(4*self.w_max/self.n_basis, 4*self.w_min/self.n_basis, self.n_basis)
        W = Variable(torch.Tensor(ws), requires_grad=True)

        # Lengths: Random between 0 and 1
        Lambdi = np.random.uniform(0, 0.01, size=self.s_dim)
        Lamb = Variable(torch.Tensor(np.diag(Lambdi)), requires_grad=True)

        # Equidistant means based on an initial observation (p.60 Deisenroth)
        init = self.env.reset()
        #mumat = np.random.multivariate_normal(init, np.diag([0.1] * self.s_dim), size=self.n_basis)
        mu_low, mu_high = init-0.1, init+0.1
        #mu_low, mu_high = self.get_mu_range()
        mumat = np.zeros((self.n_basis, self.s_dim))
        for d in range(self.s_dim):
            mumat[:, d] = np.linspace(mu_low[d], mu_high[d], self.n_basis)
        Mu = Variable(torch.Tensor(mumat), requires_grad=True)
        self.Theta = {"W": W, "Lamb": Lamb, "Mu": Mu}
        # Plot RBF policy net
        if plot_pol:
            self.plot_rbf_net()

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
        curr_x = (x-self.Theta["Mu"].detach().numpy()[i]).reshape(-1, 1)
        prod = np.dot(np.dot(np.transpose(curr_x), np.linalg.inv(self.Theta["Lamb"].detach().numpy())), curr_x)
        phi_x = np.exp(-0.5*prod).item()
        return phi_x

    def rollout(self, random=False):
        """
        Samples a traj from performing actions based on the current policy
        :param random: True, if actions are to be sampled randomly from the action space
        :return: Sampled trajs
        """
        start = timer()

        old_observation = [np.inf]*self.s_dim
        old_action = 0

        # Reset the environment
        observation = self.env.reset()
        episode_reward = 0.0
        done = False
        traj = []

        while not done:
            # Show environment
            self.env.render()
            point = []

            if not random:
                action = self.get_action(np.asarray(observation))
            else:
                action = self.env.action_space.sample()
            action = self.get_action(np.asarray(observation))
            point.append(observation)  # Save state to tuple
            point.append(action+old_action)  # Save action to tuple
            observation, reward, done, _ = self.env.step(action)  # Take action
            point.append(reward)  # Save reward to tuple

            episode_reward += reward

            # Append point if it is far enough from the previous one
            if not np.all(np.abs(observation - old_observation) < 1e-2):
                traj.append(point)  # Add Tuple to traj
                old_observation = observation
                old_action = 0
            else:
                old_action += action
                print("Sampled redundant state.")

        print("Done rollout, ", timer() - start)
        print("Episode reward: ", episode_reward)
        return traj

    def update(self, J, dJ, p):
        """
        Optimizes the policy param.s w.r.t. the expected return
        :param J: Function for computing the expected return
        :param dJ: Function for computing the gradient of the expected return
        :param p: Denotes which params are going to be optimized
        """
        init_all = self.param_array()
        init = []
        bnds = []
        # Optimizing for W
        if p==0:
            init = init_all[:self.n_basis]
            bnds = ([(self.w_min, self.w_max)] * self.n_basis)
        # Optimizing for Lambda
        elif p==1:
            init = init_all[self.n_basis:self.n_basis+self.s_dim]
            bnds = ([(1e-5, 1)] * self.s_dim)
        # Optimizing for Mu
        elif p==2:
            init = init_all[self.n_basis+self.s_dim:]
            lo, hi = self.get_mu_range()
            bnds = ([(lo[ad], hi[ad]) for ad in range(self.s_dim)] * self.n_basis)

        #new_Theta = minimize(J, init, method='L-BFGS-B', jac=dJ, bounds=bnds, options={'disp': True, 'maxfun': 1}).x
        new_Theta = minimize(J, init, method='L-BFGS-B', bounds=bnds, options={'disp': True, 'maxfun': 1}).x
        print("Optimization of policy params done.")
        new_Theta_all = init_all

        if p==0:
            new_Theta_all[:self.n_basis] = new_Theta
        elif p==1:
            new_Theta_all[self.n_basis:self.n_basis+self.s_dim] = new_Theta
        elif p==2:
            new_Theta_all[self.n_basis+self.s_dim:] = new_Theta

        self.assign_Theta(new_Theta_all)

    def check_convergence(self, old_policy):
        """
        Checks if there is a significant difference between the old and new policy param.s
        :param old_policy: Policy from previous iteration
        :return: True if convergence
        """
        new_Theta = self.param_array()
        old_Theta = old_policy.param_array()
        return np.all(np.abs(new_Theta-old_Theta) < 0.1)

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

    def plot_rbf_net(self):
        """
        Plots the RBF policy for each dimension of the observation space
        """
        for d in range(self.s_dim):
            states = []
            # Create states which range from low to high for each dimension
            for n in range(self.n_basis):
                ast = np.copy(self.Theta["Mu"].detach().numpy())[n, :]
                states.append(ast)

            # Plot features
            for i in range(self.n_basis):
                # Calc RBF values
                fun_vals_i = [self.Theta["W"].detach().numpy()[i] * self.calc_feature(state, i) for state in states]
                states_d = [s[d] for s in states]
                plt.plot(states_d, fun_vals_i)

            plt.xlabel("In")
            plt.ylabel("Out")
            plt.title("RBF Net of Policy, Dimension "+str(d))

            plt.show()

            plt.plot(range(len(states)), [self.get_action(st) for st in states])
            plt.xlabel("State Nr.")
            plt.ylabel("Chosen Action")
            plt.title("Policy Actions Based on States")
            plt.show()

            # Break because plots look same for each dim.
            break
        print("Done plotting policy net.")

    def get_mu_range(self):
        """
        Returns the highest and lowest states from the state space of the current env
        :return: Highest and lowest state
        """
        s_low = self.env.observation_space.low
        s_high = self.env.observation_space.high
        s_low[-2] = -10
        s_low[-1] = -np.pi
        s_high[-2] = 10
        s_high[-1] = np.pi
        return s_low, s_high

