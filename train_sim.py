import gym
import numpy as np
import quanser_robots
from timeit import default_timer as timer
import pickle
from matplotlib import pyplot as plt

import npg
import linear_policy
import mlp_value_function
import evaluate


def settings(env_name, delta):
    """
    Return different hyperparams based on current environment
    :param delta: If True, will return a list of deltas. Else, a list of numbers of traj. samples
    :param env_name: Name of the env
    :return: Number of iterations for plotting, delta, number of sampled trajs per iteration
    """
    num_iters, adelta, traj_samples_list = [], [], []
    if env_name == 'CartpoleStabShort-v0' or env_name == 'CartpoleStabLong-v0':
        num_iters = range(0, 151, 15)
        adelta = np.linspace(5e-4, 2e-3, 3) if delta else 0.001
        traj_samples_list = 20 if delta else [5, 10, 20]
    elif env_name == 'CartpoleSwingShort-v0' or env_name == 'CartpoleSwingLong-v0':
        num_iters = range(0, 151, 15)
        adelta = np.linspace(0.001, 0.01, 3) if delta else 0.0055
        traj_samples_list = 20 if delta else [10, 20, 30]
    elif env_name == 'BallBalancerSim-v0':
        num_iters = range(0, 201, 20)
        adelta = np.linspace(0.001, 0.01, 3) if delta else 0.01
        traj_samples_list = 50 if delta else [50, 100, 200]
    return num_iters, adelta, traj_samples_list


def train(env_names, delta=False):
    """
    Trains an agent using NPG and plots the results
    :param env_names: Different env. names
    :param delta: Whether to use learning rate or number of sampled trajectories as hyperparam.
    """
    np.random.seed(42)
    for env_name in env_names:
        num_iters, adelta, traj_samples_list = settings(env_name, delta)
        print(env_name)
        init = timer()

        all_avg, all_std = [], []
        if delta:
            alist = adelta
            msg1 = "Delta: "
            msg2 = msg1
        else:
            alist = traj_samples_list
            msg1 = "Traj. samples per iteration: "
            msg2 = " Trajectories per Iteration"

        # Loop over different numbers of trajectory samples/ different deltas
        for anitem in alist:
            if delta:
                anitem = round(anitem, 4)
            print(msg1 + str(anitem))

            avg_rewards, std_rewards = [], []

            # Setup policy, environment and models
            env = gym.make(env_name)
            env.seed(1)
            s_dim = env.observation_space.shape[0]
            policy = linear_policy.SimpleLinearPolicy(env)
            val = mlp_value_function.ValueFunction(s_dim)
            model = npg.NPG(policy, env, val, anitem if delta else adelta)

            start = timer()

            # Evaluate Model before learning with 100 rollouts
            evaluate1 = evaluate.Evaluator(policy, gym.make(env_name))
            ev_random = evaluate1.evaluate(100)
            print("Average reward+std: ", ev_random)

            print("Done evaluating init. policy, ", timer() - start)
            avg_rewards.append(ev_random[0])
            std_rewards.append(ev_random[1])

            # Loop over number of iterations
            for i in range(len(num_iters[1:])):
                iters = num_iters[i + 1] - num_iters[i]

                start = timer()

                # Train model with n Trajectories per Iteration
                model.train(iters, traj_samples_list if delta else anitem, i==0)

                print("Done training, ", timer() - start)

                start = timer()

                # Evaluate Model after learning with 100 rollouts
                evaluate2 = evaluate.Evaluator(policy, gym.make(env_name))
                ev_trained = evaluate2.evaluate(100)

                print("Done evaluating learnt policy, ", timer() - start)

                # Save current model
                pickle.dump(policy, open("policies/" + env_name + "_" + str(num_iters[i + 1]) +
                                         "_" + str(anitem) + ".slp", "wb"))

                avg_rewards.append(ev_trained[0])
                std_rewards.append(ev_trained[1])
                print(ev_trained)

            print(avg_rewards)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            all_avg.append(avg_rewards)
            all_std.append(std_rewards)

        # Plot rewards together with iterations
        plt.figure(figsize=(4.6, 3.6))
        # Plot average rewards for different numbers of trajectories/ different deltas
        for ar in range(len(all_avg)):
            avg_arr, std_arr = np.array(all_avg[ar]), np.array(all_std[ar])
            if delta:
                plt.plot(num_iters, avg_arr, label=msg2+str(round(alist[ar], 4)))
            else:
                plt.plot(num_iters, avg_arr, label=str(alist[ar])+msg2)
            plt.fill_between(num_iters, avg_arr-std_arr, avg_arr+std_arr, alpha=0.2)
        plt.xlabel("Number of Iterations")
        plt.ylabel("Average Reward on 100 Episodes")
        plt.title("Average Rewards for " + env_name)
        plt.xticks(num_iters)
        plt.legend()
        plt.tight_layout()

        plt.gcf()
        plt.savefig(("figures_delta/" if delta else "figures/") + env_name + ".pdf")

        # plt.show()
        print("Env done")
        print("Time for env: ", timer() - init)
        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
