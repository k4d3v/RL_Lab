import gym
import quanser_robots
from timeit import default_timer as timer
import pickle

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
    train_models = 1  # Number of Models that should be trained

    for i in range(train_models):

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
        ev1 = evaluate1.evaluate(100)


        end = timer()
        print("Done evaluating init. policy, ", end - start)

        start = timer()

        # Train model with 100 Iterations and 10 Trajectories per Iteration
        model.train(150, 10)

        end = timer()
        print("Done training, ", end - start)

        start = timer()

        # Evaluate Model after learning with 100 rollouts
        evaluate2 = evaluate.Evaluator(policy, gym.make(env_name))
        ev2 = evaluate2.evaluate(100)

        end = timer()
        print("Done evaluating learnt policy, ", end - start)

        # Save current model
        pickle.dump(policy, open("policies/"+env_name+"_"+str(i)+".slp", "wb"))

        avg_rewards.append([ev1, ev2])
        print([ev1, ev2])


    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(avg_rewards)
