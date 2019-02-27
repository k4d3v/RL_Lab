import gym
import quanser_robots
from timeit import default_timer as timer
import pickle
from matplotlib import pyplot as plt

import evaluate


def test(policy, env_name):
    start = timer()

    evs = []
    rng = range(10)
    for _ in rng:
        # Evaluate Model after learning with 100 rollouts
        eval = evaluate.Evaluator(policy, gym.make(env_name))
        ev = eval.evaluate(1, True, True)
        print("Total reward:", ev)
        evs.append(ev)

    end = timer()
    print("Done evaluating learnt policy, ", end - start)

    # Plot rewards together with iterations
    plt.figure(figsize=(4.6, 3.6))
    plt.plot(rng, evs)
    plt.xlabel("Episode Nr.")
    plt.ylabel("Average Reward")
    plt.title("Rewards for Repeated Experiment on Real Cartpole")
    plt.xticks(rng)
    plt.tight_layout()

    # Save plot
    plt.gcf()
    plt.savefig("figures_sim_real/" + env_name + ".pdf")


env_names = ['CartpoleStabRR-v0']
# env_names = ['CartpoleStabShort-v0']

for env_name in env_names:
    print(env_name)

    iters = [200]  # Number of iterations of training
    traj_samps = 0.0055  # Number of sampled trajs per iteration during training

    for i in iters:
        policy = pickle.load(open("policies/" + env_name + "_" + str(i) + "_" + str(traj_samps) + ".slp", "rb"))
        test(policy, env_name)
