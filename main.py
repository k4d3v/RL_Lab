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


def settings(env_name):
    """
    Return different hyperparams based on current environment
    :param env_name: Name of the env
    :return: Number of iterations for plotting, delta, Number of sampled trajs per iteration
    """
    num_iters, delta, traj_samples_list = [], 0.05, []
    if env_name == 'CartpoleStabShort-v0' or env_name == 'CartpoleStabLong-v0':
        num_iters = [0, 50, 100, 150, 200]
        delta = 0.05
        traj_samples_list = [10, 20]
    elif env_name == 'CartpoleSwingShort-v0' or env_name == 'CartpoleSwingLong-v0':
        num_iters = [0, 10, 20, 30, 40, 50]
        delta = 0.05
        traj_samples_list = [50, 80]
    elif env_name == 'BallBalancerSim-v0':
        num_iters = [0, 100, 200, 300, 400]
        delta = 0.05
        traj_samples_list = [200, 500]
    return num_iters, delta, traj_samples_list


"""
Script for testing the NPG implementation
"""
np.random.seed(42)
env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0',
             'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']

for env_name in env_names:
    num_iters, delta, traj_samples_list = settings(env_name)
    print(env_name)
    init = timer()

    all_rews = []
    # Loop over different numbers of trajectory samples
    for traj_samples in traj_samples_list:
        print("Traj. samples per iteration: "+str(traj_samples))
        avg_rewards = []

        # Setup policy, environment and models
        env = gym.make(env_name)
        s_dim = env.observation_space.shape[0]
        policy = linear_policy.SimpleLinearPolicy(s_dim)
        val = mlp_value_function.ValueFunction(s_dim)
        model = npg.NPG(policy, env, val, delta)

        start = timer()

        # Evaluate Model before learning with 100 rollouts
        evaluate1 = evaluate.Evaluator(policy, gym.make(env_name))
        ev_random = evaluate1.evaluate(100)
        print(ev_random)

        print("Done evaluating init. policy, ", timer() - start)
        avg_rewards.append(ev_random)

        # Loop over number of iterations
        for i in range(len(num_iters[1:])):
            iters = num_iters[i + 1] - num_iters[i]

            start = timer()

            # Train model with n Trajectories per Iteration
            model.train(iters, traj_samples)

            print("Done training, ", timer() - start)

            start = timer()

            # Evaluate Model after learning with 100 rollouts
            evaluate2 = evaluate.Evaluator(policy, gym.make(env_name))
            ev_trained = evaluate2.evaluate(100)

            print("Done evaluating learnt policy, ", timer() - start)

            # Save current model
            pickle.dump(policy, open("policies/" + env_name + "_" + str(num_iters[i + 1]) +
                                     "_" + str(traj_samples) + ".slp", "wb"))

            avg_rewards.append(ev_trained)
            print(ev_trained)

        print(avg_rewards)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        all_rews.append(avg_rewards)

    # Plot rewards together with iterations
    plt.figure(figsize=(4.6, 3.6))
    # Plot average rewards for different numbers of trajectories
    for ar in range(len(all_rews)):
        plt.plot(num_iters, all_rews[ar], label= str(traj_samples_list[ar]) + " Trajectories per Iteration")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average Reward on 100 Episodes")
    plt.title("Average Rewards for " + env_name)
    plt.xticks(num_iters)
    plt.legend()
    plt.tight_layout()

    plt.gcf()
    plt.savefig("figures/" + env_name + ".pdf")

    # plt.show()
    print("Env done")
    print("Time for env: ", timer() - init)
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
