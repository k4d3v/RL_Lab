import torch
import numpy as np
from timeit import default_timer as timer


class NPG:
    """
    Represents the NPG algorithm
    """

    def __init__(self, policy, env, val, delta=0.05):
        """
        Initializes NPG
        :param policy: A specific stochastic policy
        :param env: The current environment
        :param val: An approximative model for the value function
        :param delta: Normalized step size
        """
        self.policy = policy
        self.env = env
        self.s_dim = self.env.observation_space.shape[0]
        self.val = val
        self.delta = delta

    def train(self, k, n, first=False):
        """
        Trains the NPG model
        :param k: Number of iterations
        :param n: Number of sampled trajectories per iteration
        """
        # Initial fitting of V
        # Collect trajectories by rolling out the policy
        trajs = self.rollout(n)
        # Fit value function
        self.val.fit(trajs, first)

        for i in range(k):
            start = timer()
            print("Iteration: ", i)

            # Collect trajectories by rolling out the policy
            trajs = self.rollout(n)

            # Compute log-prob-gradient for each (s,a) pair of the sampled trajectories
            log_prob_grads = self.compute_grads(trajs)

            # Compute advantages
            vals = self.val.predict(trajs)
            adv = self.compute_adv(trajs, vals)

            # Compute Vanilla Policy Gradient (2)
            vanilla_gradient = self.vanilla_pol_grad(log_prob_grads, adv)

            # Compute Fisher Information Metric (4)
            fish = self.fisher(log_prob_grads)
            fish_inv = np.linalg.pinv(fish)

            # Compute gradient ascent step (5)
            step = self.grad_asc_step(vanilla_gradient, fish_inv)
            print(step)

            if all(np.abs(e) < 1e-3 for e in step):
                print("Convergence!")
                break

            # Update policy parameters
            self.policy.update_params(step.ravel())
            # print("New Params: ", self.policy.get_params())

            # Fit value function
            self.val.fit(trajs)

            end = timer()
            print("Done iteration, ", end - start)

        print("Finished training")

    def grad_asc_step(self, vanilla_gradient, fisher_inv):
        """
        Computes Theta_k for gradient ascent as in (5)
        :param vanilla_gradient: The policy gradient
        :param fisher_inv: The inverted Fisher matrix
        :return: Theta_k as in (5)
        """
        start = timer()

        alpha = np.sqrt(self.delta / (np.matmul(np.matmul(vanilla_gradient, fisher_inv), vanilla_gradient.T)))
        nat_grad = np.matmul(fisher_inv, vanilla_gradient.T)

        end = timer()
        print("Done grad asc step, ", end - start)
        return alpha * nat_grad

    def fisher(self, log_prob_grads):
        """
        Computes the Fisher information metric (4)
        :param log_prob_grads:
        :return: Fisher matrix
        """
        start = timer()

        num_trajs = len(log_prob_grads)  # Number of trajectories
        num_grad = len(log_prob_grads[0][0])  # Dimension of Gradient

        f = np.zeros((num_grad, num_grad))
        for i in range(num_trajs):
            num_timesteps = len(log_prob_grads[i])

            traj_f = np.zeros((num_grad, num_grad))
            for j in range(num_timesteps):
                traj_f += np.outer(log_prob_grads[i][j], log_prob_grads[i][j])

            traj_f /= num_timesteps
            f += traj_f

        f /= num_trajs

        end = timer()
        print("Done Fisher, ", end - start)
        return f

    def vanilla_pol_grad(self, log_prob_grads, adv):
        """
        Computes the policy gradient as in (2)
        :param log_prob_grads: The log pi for every state-action pair along trajs
        :param adv: Estimated advantages based on approximated value fun
        :return: The policy gradient
        """
        start = timer()

        num_trajs = len(log_prob_grads)  # Number of trajectories
        num_grad = len(log_prob_grads[0][0])  # Dimension of Gradient

        pol_grad = np.zeros((1, num_grad))
        for i in range(num_trajs):
            num_timesteps = len(log_prob_grads[i])

            traj_pol_grad = np.zeros((1, num_grad))
            for j in range(num_timesteps):
                traj_pol_grad += log_prob_grads[i][j] * adv[i][j]

            traj_pol_grad /= num_timesteps
            pol_grad += traj_pol_grad

        pol_grad /= num_trajs

        end = timer()
        print("Done vanilla, ", end - start)
        return pol_grad

    def compute_grads(self, trajs):
        """
        Computes the gradient of log pi
        :param trajs: Sampled trajs
        :return: The gradient of log pi for each (s,a) pair along the trajs
        """
        start = timer()

        all_grads = []
        for traj in trajs:
            traj_grads = []
            for timestep in traj:
                obs, act, _ = timestep
                grad = self.policy.get_gradient(torch.Tensor(obs).view(self.s_dim, 1), torch.from_numpy(act.ravel()))
                #grad = self.policy.get_gradient_analy(torch.Tensor(obs).view(self.s_dim, 1),
                #                                      torch.from_numpy(act.ravel()))
                traj_grads.append(grad)
            all_grads.append(traj_grads)

        end = timer()
        print("Done log grads, ", end - start)
        return all_grads

    def compute_adv(self, trajs, vals, gamma=0.95, lamb=0.97):
        """
        Computes the advantages based on the sampled trajs
        See High-Dimensional Continuous Control Using Generalized Advantage Estimation, p.5
        :param trajs: Sampled trajs
        :param vals: Approximated value fun
        :param gamma: Gamma hyperparam
        :param lamb: Lambda hyperparam
        :return: Estimated advantages
        """
        start = timer()

        all_adv = []

        trajs_rew, gavals = [], []
        for i in range(len(trajs)):
            # Get rewards from data
            trajs_rew.append(np.array([p[2] for p in trajs[i]][:-1]))

            # Compute gamma*v for each point on the trajectory and shift the vector one to the left
            gavals.append(np.array([gamma * v for v in vals[i][1:]]).reshape(-1, ))

            # Reshape vals
            vals[i] = np.array(vals[i][:-1]).reshape(-1, )

        for i in range(len(trajs)):
            # traj + gamma*vals - vals
            curr_sum = np.subtract(np.add(trajs_rew[i], gavals[i]), vals[i])

            adv_row = []
            for t in range(len(trajs[i])):
                # Sum up all entries delta, multiplied by (gamma*lambda)^l, starting at j
                l_sum = curr_sum[t:]
                l_sum = [((gamma * lamb) ** l) * l_sum[l] for l in range(l_sum.shape[0])]
                adv_row.append(np.sum(l_sum))

            # Append advantages for each trajectory
            all_adv.append(adv_row)

        end = timer()
        print("Done advantas, ", end - start)
        return all_adv

    def rollout(self, n):
        """
        Samples some trajs from performing actions based on the current policy
        :param n: Number of trajs
        :return: Sampled trajs
        """
        start = timer()

        trajs = []
        avg_reward = 0.0

        for _ in range(n):
            # Reset the environment
            observation = self.env.reset()
            episode_reward = 0.0
            done = False
            traj = []

            while not done:
                #self.env.render()
                point = []

                # Clip action if on real env
                action = np.clip(self.policy.get_action(torch.Tensor(observation).view(self.s_dim, 1)), -6, 6) \
                    if self.env.spec.id == "CartpoleStabRR-v0" \
                    else self.policy.get_action(torch.Tensor(observation).view(self.s_dim, 1))

                point.append(observation)  # Save state to tuple
                point.append(action)  # Save action to tuple
                observation, reward, done, _ = self.env.step(action)  # Take action
                point.append(reward)  # Save reward to tuple

                episode_reward += reward
                traj.append(point)  # Add Tuple to traj

                if self.env.spec.id == "CartpoleStabRR-v0":
                    min_s0, max_s0 = self.env.observation_space.low[0], self.env.observation_space.high[0]
                    distl = np.abs(observation[0]-min_s0)
                    distr = np.abs(observation[0]-max_s0)
                    #print(distl)
                    #print(distr)
                    if distl<0.1 or distr<0.1:
                        break

            # Delete out of bounds (last) point on traj for ballbal
            if self.env.spec.id == "BallBalancerSim-v0":
                del traj[-1]
            avg_reward += episode_reward
            #print(len(traj))
            #trajs.append(self.clean(traj))
            trajs.append(traj)

        avg_reward /= n
        print("Avg reward: ", avg_reward)
        end = timer()
        print("Done rollout, ", end - start)
        self.recent_reward = avg_reward

        self.env.step(np.array([0.]))
        return trajs

    def clean(self, traj):
        """
        Removes redundant samples from a trajectory
        :param traj: Trajectory
        :return: Cleant trajectory
        """
        t = 1
        while t < len(traj):
            if np.all(np.abs(traj[t][0] - traj[t - 1][0]) < 5e-3) and np.abs(traj[t][2]-traj[t-1][2]) < 1e-3:
                # Remove previous sample
                del traj[t - 1]
            else:
                t += 1
        return traj
