import random

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

"""
Script for testing the NPG implementation
"""

random.seed(42)
env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0', 'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']
#env_names = ['BallBalancerSim-v0']

num_iters = [0, 50, 100, 150, 200, 250]  # Different numbers of iterations

deltas = [0.05]*5
traj_samples_list = [20, 20, 100, 100, 200] # TODO: Finetune

for env_name, delta, traj_samples in zip(env_names, deltas, traj_samples_list):
    print(env_name)
    init = timer()

    avg_rewards = []

    print("########################################")
    # Setup policy, environment and models
    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    policy = linear_policy.SimpleLinearPolicy(s_dim) # TODO: Maybe try RBF policy
    val = mlp_value_function.ValueFunction(s_dim)
    model = npg.NPG(policy, env, val, delta)

    start = timer()

    # Evaluate Model before learning with 100 rollouts
    evaluate1 = evaluate.Evaluator(policy, gym.make(env_name))
    ev_random = evaluate1.evaluate(100)
    print(ev_random)

    print("Done evaluating init. policy, ", timer() - start)
    avg_rewards.append(ev_random)

    for i in range(len(num_iters[1:])):
        iters = num_iters[i+1] - num_iters[i]

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
        pickle.dump(policy, open("policies/"+env_name+"_"+str(num_iters[i+1])+".slp", "wb"))

        avg_rewards.append(ev_trained)
        print(ev_trained)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(avg_rewards)

    # Plot rewards together with iterations
    plt.figure(figsize=(4.6, 3.6))
    plt.plot(num_iters, avg_rewards)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average Reward on 100 Episodes")
    plt.title("Average Rewards for "+env_name)
    plt.xticks(num_iters)
    plt.tight_layout()

    plt.gcf()
    plt.savefig("figures/"+env_name+".pdf")

    #plt.show()
    print("Env done")
    print("Time for env: ", timer()-init)
