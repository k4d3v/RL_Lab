""" Trains with different state and action space discretizations and compares the results"""

import pickle
import gym
import quanser_robots
import matplotlib.pyplot as plt

from dyn_prog import DynProg
from policy import RandomExplorationPolicy


def compare_rewards(env_name, discrets):
    """
    Trains via Value and Policy Iteration on different discretizations and plots cumulative rewards
    :param env_name: Name of the current environment
    :param discrets: Contains different numbers of discrete entries in the state and action space
    (Each has to return an integer when sqrted!)
    """
    print(env_name)
    # Open dynamics and reward NNs
    reward = pickle.load(open("nets/rew_" + env_name + ".fitnn", 'rb'))
    dynamics = pickle.load(open("nets/dyn_" + env_name + ".fitnn", "rb"))

    env = gym.make(env_name)
    policy = RandomExplorationPolicy()

    for discret in discrets:
        print("Number of discrete entries: ", discret)
        # Train agent
        agent = DynProg(policy, env, reward, dynamics, discret)
        Vk1, pol1, cumul_rew1 = agent.train_val_iter()
        #Vk2, pol2, cumul_rew2 = agent.train_pol_iter()

        # Make it pretty!
        plt.plot(discrets, cumul_rew1, label="Value iteration")
        #plt.plot(discrets, cumul_rew2, label="Policy iteration")
        plt.title("Comparison of learning for different dicretizations")
        plt.xlabel("Number of discrete entries in the state and action space")
        plt.ylabel("Cumulative reward after training")
        plt.legend()

        # Save plot
        fig = plt.gcf()
        fig.savefig('figures/cumul_rew_comparison_' + env_name + '.png')

        # Show plot
        plt.show()


# Pendulum
compare_rewards("Pendulum-v2", [64])

# Qube
# compare_rewards("Qube-v0", 25000)

#TODO: Plot best results of Value fun and Policy on Pendulum-v2