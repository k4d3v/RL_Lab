import numpy as np
from scipy.optimize import minimize

class LinearPolicy:
    """
    Linear policy with weights Psi and an offset v
    See https://publikationen.bibliothek.kit.edu/1000019799
    """
    def __init__(self, env):
        """
        :param env: Environment
        """
        super().__init__()
        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.a_max = self.env.action_space.high
        self.n_params = 2
        self.Psi = np.random.normal(0, 0.1, size=(self.a_dim, self.s_dim))
        self.v = np.random.normal(0, 0.1, size=(self.a_dim, 1))

    def get_action(self, x, raw=False):
        """
        Returns a single control based on observation x
        :param x: Observation
        :param raw: If true, an unsquashed control will be returned
        :return: Control
        """
        a_raw = np.add(np.dot(self.Psi, x), self.v).reshape(self.a_dim, )
        return a_raw if raw else self.a_max*np.sin(a_raw)

    def rollout(self, random=False):
        """
        Samples a traj from performing actions based on the current policy
        :param random: True, if actions are to be sampled randomly from the action space
        :return: Sampled traj
        """
        # Reset the environment
        observation = self.env.reset()

        episode_reward = 0.0
        done = False
        traj = []

        while not done:
            # Show environment
            #self.env.render()
            point = []

            if not random:
                action = self.get_action(np.asarray(observation))
            # Apply random actions for initial rollout
            else:
                action = self.env.action_space.sample()
            # Clip controls for cartpole
            if self.env.env.spec.id != "BallBalancer-v0":
                action = np.clip(action, -6, 6)
            #print(action)

            point.append(observation)  # Save state to tuple
            point.append(action)  # Save action to tuple
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
        pl = [self.Psi, self.v]
        params = [list(param.flatten()) for param in pl]
        return np.array(params[0] + params[1])

    def assign_Theta(self, params):
        """
        Assign Psi and v
        :param params: List containing Psi and v
        """
        self.Psi = params[:self.s_dim * self.a_dim].reshape(self.a_dim, self.s_dim)
        self.v = params[self.s_dim * self.a_dim:].reshape(self.a_dim, 1)

    def update(self, J, p):
        """
        Optimizes the policy param.s w.r.t. the expected return
        :param J: Function for computing the expected return
        :param p: Denotes which params are going to be optimized
        """
        init_all = self.param_array()
        init, bnds = [], []

        # Optimizing for Psi
        if p == 0:
            init = init_all[:self.a_dim * self.s_dim]
            # Bounds Psi to (-1,1)
            bnds = ([(-1, 1) for _ in range(self.s_dim)] * self.a_dim)
        # Optimizing for v
        elif p == 1:
            init = init_all[self.a_dim * self.s_dim:]
            # Bounds offset to (-1,1)
            bnds = ([(-1, 1)] * self.a_dim)
        # Joint optimization
        elif p == -1:
            init = init_all
            # Bounds centers at the state boundary + min. and max action
            bnds = ([(-1, 1) for _ in range(self.s_dim)] * self.a_dim + [(-1, 1)] * self.a_dim)

        new_Theta = minimize(J, init, method='L-BFGS-B', bounds=bnds, options={'disp': True, 'maxfun': 1}).x
        print("Optimization of policy params done.")
        new_Theta_all = init_all

        if p == 0:
            new_Theta_all[:self.s_dim * self.a_dim] = new_Theta
        elif p == 1:
            new_Theta_all[self.s_dim * self.a_dim:] = new_Theta
        elif p == -1:
            new_Theta_all[:] = new_Theta

        self.assign_Theta(new_Theta_all)

    def check_convergence(self, old_policy):
        """
        Checks if there is a significant difference between the old and new policy param.s
        :param old_policy: Policy from previous iteration
        :return: True if convergence
        """
        new_Theta = self.param_array()
        old_Theta = old_policy.param_array()
        return np.all(np.abs(new_Theta - old_Theta) < 0.1)

    def pred_distro(self, mu, Sigma):
        """
        Computes the predictive distribution for a mu and Sigma
        :param mu: Mean of state distribution
        :param Sigma: Variance of state distro
        :return: mean and covariance of predictive distro
        """
        E = self.get_action(mu, True)
        cov = np.dot(np.dot(self.Psi, Sigma), self.Psi.T)
        return E, cov
