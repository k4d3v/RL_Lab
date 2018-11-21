import gym
import quanser_robots

import npg
import linear_policy
import mlp_value_function
import evaluate

"""
Script for testing the NPG implementation
"""

avg_rewards = []
train_models = 1  # Number of Models that should be trained

for _ in range(train_models):

    print("########################################")
    # Setup policy, environment and models
    policy = linear_policy.SimpleLinearPolicy()
    env = gym.make('CartpoleStabShort-v0')
    val = mlp_value_function.ValueFunction()
    model = npg.NPG(policy, env, val)

    # Evaluate Model before learning with 100 rollouts
    evaluate1 = evaluate.Evaluator(policy, gym.make('CartpoleStabShort-v0'))
    ev1 = evaluate1.evaluate(100)

    # Train model with 100 Iterations and 10 Trajectories per Iteration
    model.train(100, 10)

    # Evaluate Model after learning with 100 rollouts
    evaluate2 = evaluate.Evaluator(policy, gym.make('CartpoleStabShort-v0'))
    ev2 = evaluate2.evaluate(100)

    avg_rewards.append([ev1, ev2])
    print([ev1, ev2])


print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(avg_rewards)
