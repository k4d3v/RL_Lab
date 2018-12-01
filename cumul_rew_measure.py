""" Trains with different state and action space discretizations and compares the results"""
import random

import numpy as np
import pickle
import gym
import quanser_robots
import matplotlib.pyplot as plt
import torch

from dyn_prog import DynProg
from policy import RandomExplorationPolicy


def plot_results(algo_name, env_name, cumul_rew_list, n_states, n_acts):
    """
    Plots cumulative rewards for a given algo for different state and action space discretizations
    :param algo_name: Name of the algorithm
    :param env_name: Name of the environment
    :param cumul_rew_list:
    :param n_acts: List with different numbers of states
    :param n_acts: List with different numbers of actions
    """
    # Iterate over number of actions and plot curves with values for different state numbers
    for na in range(len(cumul_rew_list)):
        plt.plot(n_states, cumul_rew_list[na], label=str(n_acts[na]) + " actions")
    plt.title("Comparison of " + algo_name + " results for different discretizations on "+env_name)
    plt.xlabel("Number of discrete entries in the state space")
    plt.ylabel("Cumulative reward after training")
    plt.legend()

    # Save plot
    #fig = plt.gcf()
    #fig.savefig('figures/cumul_rew_' + algo_name + '_comparison_' + env_name + '.png')

    # Show plot
    plt.show()


def compare_rewards(env_name, n_states, n_acts):
    """
    Trains via Value and Policy Iteration on different discretizations and plots cumulative rewards
    :param env_name: Name of the current environment
    :param n_states: List with state space discretizations
    :param n_acts: List with action space discretizations
    """
    print(env_name)
    # Open dynamics and reward NNs
    reward = pickle.load(open("nets/rew_" + env_name + ".fitnn", 'rb'))
    dynamics = pickle.load(open("nets/dyn_" + env_name + ".fitnn", "rb"))

    env = gym.make(env_name)
    print("Reward range: ", env.reward_range)
    policy = RandomExplorationPolicy()

    cumul_rew_list1, cumul_rew_list2 = [], []

    for na in n_acts:
        print("Number of discrete actions: ", na)
        cumul_rew_states1, cumul_rew_states2 = [], []
        for ns in n_states:
            print("Number of discrete states: ", ns)

            # Train agent
            agent = DynProg(policy, env, reward, dynamics, (ns, na))
            _, cumul_rew1 = agent.train_val_iter()
            cumul_rew_states1.append(cumul_rew1)

            # Vk2, pol2, cumul_rew2 = agent.train_pol_iter()

        cumul_rew_list1.append(cumul_rew_states1)
        # cumul_rew_list2.append(cumul_rew_states2)

    # Value iteration plot
    plot_results("value_iteration", env_name, cumul_rew_list1, n_states, n_acts)

    # Policy iteration plot
    #plot_results("policy_iteration", env_name, cumul_rew_list2)


def compare_rewards_value_iteration(env_name, n_states, n_acts):
    """
    Trains via Value and Policy Iteration on different discretizations and plots cumulative rewards
    :param env_name: Name of the current environment
    :param n_states: List with state space discretizations
    :param n_acts: List with action space discretizations
    """
    print(env_name)
    # Open dynamics and reward NNs
    reward = pickle.load(open("nets/rew_" + env_name + ".fitnn", 'rb'))
    dynamics = pickle.load(open("nets/dyn_" + env_name + ".fitnn", "rb"))

    env = gym.make(env_name)
    policy = RandomExplorationPolicy()

    cumul_rew_list1 = []

    for na in n_acts:
        print("Number of discrete actions: ", na)
        cumul_rew_states1, cumul_rew_states2 = [], []
        for ns in n_states:
            print("Number of discrete states: ", ns)

            # Train agent
            agent = DynProg(policy, env, reward, dynamics, (ns, na))
            _, pol = agent.train_val_iter()


            # Evaluate Policy
            val_env = gym.make(env_name)
            states, actions = pol
            total_reward = 0.0
            for i in range(100):

                # Reset the environment
                observation = val_env.reset()
                done = False

                episode_reward = 0.0
                while not done:
                    idx = ((states - observation) ** 2).sum(1).argmin()
                    action = [actions[idx]]
                    observation, rew, done, _ = val_env.step(action)  # Take action
                    episode_reward += rew

                total_reward += episode_reward

            total_reward /= 100.0
            cumul_rew_states1.append(total_reward)

        cumul_rew_list1.append(cumul_rew_states1)

    # Value iteration plot
    plot_results("value_iteration", env_name, cumul_rew_list1, n_states, n_acts)

# Pendulum
#compare_rewards("Pendulum-v2", [100, 400, 900, 1600, 2500], [4, 8, 16, 32])
#compare_rewards("Pendulum-v2", [4, 9, 16, 25], [2, 4, 8])

# Qube
#compare_rewards("Qube-v0", [100, 400, 900, 1600, 2500], [4, 8, 16, 32])

compare_rewards_value_iteration("Pendulum-v2", [1600, 2500, 3600, 4900, 6400], [100])
