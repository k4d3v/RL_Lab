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
Script for training a model on the real system after having pretrained in simulation
"""
np.random.seed(42)
#env_name = 'CartpoleStabRR-v0'
env_name = 'CartpoleStabShort-v0'

num_iters, delta, traj_samples = range(10), 0.0055, 5
iter_simu = 200

print(env_name)
init = timer()

avg_rewards = []

# Setup policy, environment and models
env = gym.make(env_name)
env.seed(1)
s_dim = env.observation_space.shape[0]
# Load policy learnt in simulation
policy = pickle.load(open("policies/" + env_name + "_" + str(iter_simu) + "_" + str(delta) + ".slp", "rb"))
val = mlp_value_function.ValueFunction(s_dim)
model = npg.NPG(policy, env, val, delta)

# Evaluate Model simulation on 100 rollouts before learning
start = timer()
sim_env = gym.make('CartpoleStabShort-v0')
sim_env.seed(1)
evaluate1 = evaluate.Evaluator(policy, sim_env)
ev_random = evaluate1.evaluate(100)
print(ev_random)

print("Done evaluating init. policy, ", timer() - start)
avg_rewards.append(ev_random)

# Loop over number of iterations
for i in range(len(num_iters[1:])):
    iters = num_iters[i + 1] - num_iters[i]

    # Train model with n Trajectories per Iteration
    start = timer()
    model.train(iters, traj_samples, i == 0)
    # Save reward after each iteration
    avg_rewards.append(model.recent_reward)
    print("Done training, ", timer() - start)

# Save model after learning
pickle.dump(policy, open("policies/" + env_name + "_" + str(iter_simu) +
                         "_" + str(delta) + "_real.slp", "wb"))

# Plot rewards together with iterations
print(avg_rewards)
plt.figure(figsize=(4.6, 3.6))
plt.plot(num_iters, avg_rewards)
plt.xlabel("Number of Iterations")
plt.ylabel("Average Reward on " + str(traj_samples) + " Episodes")
plt.title("Average Rewards for " + env_name)
plt.xticks(num_iters)
plt.tight_layout()

plt.gcf()
plt.savefig("figures_real_real/" + env_name + ".pdf")

# plt.show()
print("Env done")
print("Time for env: ", timer() - init)