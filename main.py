import gym
import quanser_robots
from timeit import default_timer as timer
import pickle
from matplotlib import pyplot as plt

import npg
import linear_policy_new as linear_policy
import mlp_value_function
import evaluate

"""
Script for testing the NPG implementation
"""

env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0', 'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']

for env_name in env_names:
    print(env_name)

    avg_rewards = []
    num_iters = [0, 50, 100, 150, 200]  # Different numbers of iterations

    print("########################################")
    # Setup policy, environment and models
    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    policy = linear_policy.SimpleLinearPolicy(s_dim) # TODO: Maybe try RBF policy
    val = mlp_value_function.ValueFunction(s_dim)
    model = npg.NPG(policy, env, val)

    start = timer()

    # Evaluate Model before learning with 100 rollouts
    evaluate1 = evaluate.Evaluator(policy, gym.make(env_name))
    ev_random = evaluate1.evaluate(100)
    print(ev_random)

    end = timer()
    print("Done evaluating init. policy, ", end - start)
    avg_rewards.append(ev_random)

    for i in range(len(num_iters[1:])):
        iters = num_iters[i+1] - num_iters[i]

        start = timer()

        # Train model with 10 Trajectories per Iteration
        model.train(iters, 10)

        end = timer()
        print("Done training, ", end - start)

        start = timer()

        # Evaluate Model after learning with 100 rollouts
        evaluate2 = evaluate.Evaluator(policy, gym.make(env_name))
        ev_trained = evaluate2.evaluate(100)

        end = timer()
        print("Done evaluating learnt policy, ", end - start)

        # Save current model
        pickle.dump(policy, open("policies/"+env_name+"_"+str(num_iters[i+1])+".slp", "wb"))

        avg_rewards.append(ev_trained)
        print(ev_trained)

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(avg_rewards)

    # Plot rewards together with iterations
    plt.plot(num_iters, avg_rewards)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Average Reward on 100 Episodes")
    plt.title("Average Reward over Iterations, "+env_name)
    plt.locator_params(axis='x', nbins=len(num_iters))
    plt.show()
    print("Env done")
