import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class GPPolicy:
    """
    Gaussian Process with a squared exponential kernel for learning a policy.
    See https://publikationen.bibliothek.kit.edu/1000019799
    """

    def __init__(self, env, n_basis=50):
        """
        :param env: Environment
        :param n_basis: Number of basis funs
        """
        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.a_max = self.env.action_space.high
        self.n_basis = n_basis
        # self.n_params = 2
        self.n_params = 1
        self.x, self.y = None, None

    def prepare_data(self, x0):
        """
        Initializes the policy parameters (test inputs and targets) (see Deisenroth p. 60)
        :param x0: Initial observation
        """
        x = np.random.multivariate_normal(x0, np.diag(np.array([0.1] * self.s_dim)), self.n_basis)
        y = np.random.normal(0, 0.1, self.n_basis)
        return x, y

    def fit_gp(self):
        """
        Fits a sklearn GP based on the training data
        """
        kern = ConstantKernel(1 ** 2, constant_value_bounds=(1, 9)) \
               * RBF(length_scale=[1] * self.s_dim, length_scale_bounds=np.array([0.2, 1])) \
               + WhiteKernel(noise_level=1e-5)
        # gp = GaussianProcessRegressor(kernel=kern, optimizer=None)
        gp = GaussianProcessRegressor(kernel=kern, n_restarts_optimizer=10)
        gp.fit(self.x, self.y)
        # self.lambs = gp.kernel_.get_params()["length_scale"]
        opti_params = gp.kernel_.get_params()
        self.alpha = np.sqrt(opti_params["k1"].k1.constant_value)
        self.lambs = opti_params["k1"].k2.length_scale
        self.noise = opti_params["k2"].noise_level
        self.gp = gp
        # print("GPML kernel: %s" % gp.kernel_)
        # print(gp.kernel_.get_params())
        print("Done fitting GP")

    def get_action(self, x, raw=False):
        """
        Returns a single control based on observation x
        :param x: Observation
        :param raw: If True, the control signal want be squashed
        :return: Control
        """
        a_raw = self.gp.predict([x])
        return a_raw if raw else self.a_max * np.sin(a_raw)

    def assign_Theta(self, params):
        """
        Assign pseudo test inputs and targets
        :param params: List containing x and y values for fitting
        """
        self.x = params[:self.s_dim * self.n_basis].reshape(self.n_basis, self.s_dim)
        self.y = params[self.s_dim * self.n_basis:]
        self.fit_gp()

    def rollout(self, random=False):
        """
        Samples a traj from performing actions based on the current policy
        :param random: True, if actions are to be sampled randomly from the action space
        :return: Sampled traj
        """
        # Reset the environment
        observation = self.env.reset()

        if random and self.x is None:
            # Generate random params and fit GP upon first rollout
            self.x, self.y = self.prepare_data(observation)
            self.fit_gp()
            self.plot_policy()

        episode_reward = 0.0
        done = False
        traj = []

        while not done:
            # Show environment
            # self.env.render()
            point = []

            if not random:
                action = self.get_action(np.asarray(observation))
            # Apply random actions for initial rollout
            else:
                action = self.env.action_space.sample()
            # Clip controls for cartpole
            if self.env.env.spec.id != "BallBalancer-v0":
                action = np.clip(action, -6, 6)
            # print(action)

            point.append(observation)  # Save state to tuple
            point.append(action)  # Save action to tuple
            # point.append(action + old_action)
            observation, reward, done, _ = self.env.step(action)  # Take action
            point.append(reward)  # Save reward to tuple

            episode_reward += reward
            traj.append(point)

        print("Episode reward: ", episode_reward)
        return traj

    def param_array(self):
        """
        Returns an array containing all the policy parameters
        :return: np array with the policy params
        """
        pl = [self.x, self.y]
        params = [list(param.flatten()) for param in pl]
        return np.array(params[0] + params[1])

    def update(self, J, p):
        """
        Optimizes the policy param.s w.r.t. the expected return
        :param J: Function for computing the expected return
        :param p: Denotes which params are going to be optimized
        """
        init_all = self.param_array()
        init, bnds = [], []

        # Store some vars
        s_low = self.env.observation_space.low
        s_high = self.env.observation_space.high
        s_low[-2] = -10
        s_low[-1] = -np.pi
        s_high[-2] = 10
        s_high[-1] = np.pi
        a_min = -np.arcsin(1)
        a_max = np.arcsin(1)

        # Optimizing for inputs
        if p == 0:
            init = init_all[:self.n_basis * self.s_dim]
            # Bounds centers at the state boundary
            bnds = ([(lowd, highd) for lowd, highd in zip(s_low, s_high)] * self.n_basis)
        # Optimizing for targets
        elif p == 1:
            init = init_all[self.n_basis * self.s_dim:]
            # Bounds centers at the minimal and maximal action
            bnds = ([(a_min, a_max)] * self.n_basis)
        # Joint optimization
        elif p == -1:
            init = init_all
            # Bounds centers at the state boundary + min. and max action
            bnds = ([(lowd, highd) for lowd, highd in zip(s_low, s_high)] * self.n_basis
                    + [(a_min, a_max)] * self.n_basis)

        new_Theta = minimize(J, init, method='L-BFGS-B', bounds=bnds, options={'disp': True, 'maxfun': 1}).x
        print("Optimization of policy params done.")
        new_Theta_all = init_all

        print("Old params: ", init)
        if p == 0:
            new_Theta_all[:self.s_dim * self.n_basis] = new_Theta
        elif p == 1:
            new_Theta_all[self.s_dim * self.n_basis:] = new_Theta
        elif p == -1:
            new_Theta_all[:] = new_Theta
        print("New params: ", new_Theta)

        self.assign_Theta(new_Theta_all)

    def check_convergence(self, old_policy):
        """
        Checks if there is a significant difference between the old and new policy param.s
        :param old_policy: Policy from previous iteration
        :return: True if convergence
        """
        new_Theta = self.param_array()
        old_Theta = old_policy.param_array()
        return np.all(np.abs(new_Theta - old_Theta) < 1e-3)

    def plot_policy(self):
        """
        Plots the current policy
        """
        plt.plot(range(len(self.x)), self.y)
        plt.title("Current Policy")
        plt.xlabel("State Nr.")
        plt.ylabel("Action")
        plt.show()
        print("Done plotting policy")
