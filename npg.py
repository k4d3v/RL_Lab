import torch
import numpy as np


class NPG:
    def __init__(self, policy, env, val):
        self.policy = policy
        self.env = env
        self.val = val

    def train(self, k, n):

        for i in range(k):
            #print("------------------------")
            print("Iteration: ", i)

            # Collect trajectories by rolling out the policy
            trajs = self.rollout(n)

            # Compute log-prob-gradient for each (s,a) pair of the sampled trajectories
            log_prob_grads = self.compute_grads(trajs)

            # Compute advantages
            vals = self.val.predict(trajs)
            adv = self.compute_adv(trajs, vals)

            # Compute Vanilla Policy Gradient
            vanilla_gradient = self.vanilla_pol_grad(log_prob_grads, adv)

            # Compute Fisher Information Metric
            fish = self.fisher(log_prob_grads)
            fish_inv = np.linalg.inv(fish)

            # Compute gradient ascent step
            step = self.grad_asc_step(vanilla_gradient, fish_inv)

            # Update policy parameters
            self.policy.update_params(step.ravel())
            #print("New Params: ", self.policy.get_params())

            # Fit value function
            self.val.fit(trajs)

    def grad_asc_step(self, vanilla_gradient, fisher_inv, delta=0.01):
        alpha = np.sqrt(delta / (np.matmul(np.matmul(vanilla_gradient, fisher_inv), vanilla_gradient.T)))
        nat_grad = np.matmul(fisher_inv, vanilla_gradient.T)
        return alpha * nat_grad

    def fisher(self, log_prob_grads):
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
        return f

    def vanilla_pol_grad(self, log_prob_grads, adv):
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
        return pol_grad

    def compute_grads(self, trajs):
        all_grads = []
        for traj in trajs:
            traj_grads = []
            for timestep in traj:
                obs, act, _ = timestep
                #grad = self.policy.get_gradient(torch.Tensor(obs).view(5, 1), torch.from_numpy(act.ravel()))
                grad  = self.policy.get_gradient_analy(torch.Tensor(obs).view(5, 1), torch.from_numpy(act.ravel()))
                traj_grads.append(grad)
            all_grads.append(traj_grads)
        return all_grads

    def compute_adv(self, trajs, vals, gamma=0.95, lamb=0.97):
        all_adv = []
        for i in range(len(trajs)):
            traj_adv = []
            for j in range(len(trajs[i])):
                adv = 0.0
                for l in range(len(trajs[i])-j-1):
                    delta = trajs[i][j + l][2] + gamma * vals[i][j + l + 1] - vals[i][j + l]
                    adv += ((gamma*lamb)**l)*delta
                traj_adv.append(adv)
            all_adv.append(traj_adv)
        return all_adv

    def rollout(self, n):
        trajs = []
        avg_reward = 0.0

        for _ in range(n):

            # Reset the environment
            observation = self.env.reset()
            episode_reward = 0.0
            done = False
            traj = []

            while not done:
                # env.render()
                point = []

                action = self.policy.get_action(torch.Tensor(observation).view(5, 1))

                point.append(observation)  # Save state to tuple
                point.append(action)  # Save action to tuple
                observation, reward, done, _ = self.env.step(action)  # Take action
                point.append(reward)  # Save reward to tuple

                episode_reward += reward
                traj.append(point)  # Add Tuple to traj

            avg_reward += episode_reward
            trajs.append(traj)

        #print("Avg reward: ", (avg_reward / n))
        return trajs
