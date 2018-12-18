""" Trains with different state and action space discretizations and compares the results"""
import random

import numpy as np
import pickle
import gym
import quanser_robots
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import torch

from dyn_prog import DynProg
from build_model import load_model


def pick_color_act(act):
    """
    Chooses a color depending on action
    :param act: Action
    :return: Color string
    """
    if act > 1:
        return "red"
    elif act > 0:
        return "orange"
    elif act < -1:
        return "blue"
    elif act < 0:
        return "cyan"


def plot_results(algo_name, env_name, cumul_rew_list, n_states, n_acts):
    """
    Plots cumulative rewards for a given algo for different state and action space discretizations
    :param algo_name: Name of the algorithm
    :param env_name: Name of the environment
    :param cumul_rew_list:
    :param n_states: List with different numbers of states
    :param n_acts: List with different numbers of actions
    """
    # Plot random results
    global rand_rews
    for ai, i in zip(n_acts, range(len(n_acts))):
        plt.plot(n_states, rand_rews[i], label="Baseline of random policy, " + str(ai) + " actions")

    # Iterate over number of actions and plot curves with values for different state numbers
    for na in range(len(cumul_rew_list)):
        plt.plot(n_states, cumul_rew_list[na], label=str(n_acts[na]) + " actions")
    plt.title("Comparison of " + algo_name + " results for different discretizations on " + env_name)
    plt.xlabel("Number of discrete entries in the state space")
    plt.ylabel("Cumulative reward after training")
    plt.legend()

    # Save plot
    fig = plt.gcf()
    fig.savefig('figures/cumul_rew_' + algo_name + '_comparison_' + env_name + '.png')

    # Show plot
    plt.show()


def plot_pol(states, actions):
    """
    Plots a policy
    :param states: States of model
    :param actions: Optimal actions for each state
    :return:
    """
    # Plot samples
    plt.scatter([s[0] for s in states], [s[1] for s in states], c=[pick_color_act(act) for act in actions])
    plt.title("Learnt policy (blue: -, red: +)")
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.show()


def evaluate_policy(env_name, pol, agent=None):
    """
    Computes the cumulative reward for a given policy on a 100 episode rollout
    :param env_name: Name of the environment
    :param pol: Contains space discretization and optimal actions
    :return: Total reward
    """
    val_env = gym.make(env_name)
    states, actions = pol

    if agent is not None:
        plot_pol(states, actions)

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

    total_reward /= 100
    print("Total reward: ", total_reward)
    return total_reward


def compare_rewards(env_name, n_states, n_acts):
    """
    Trains via Value and Policy Iteration on different discretizations and plots cumulative rewards
    :param env_name: Name of the current environment
    :param n_states: List with state space discretizations
    :param n_acts: List with action space discretizations
    """
    print(env_name)

    cumul_rew_list1, cumul_rew_list2 = [], []

    for na in n_acts:
        print("Number of discrete actions: ", na)
        cumul_rew_states1, cumul_rew_states2 = [], []
        for ns in n_states:
            print("Number of discrete states: ", ns)

            # Load model
            model = load_model(env_name, (ns+int(ns/100), na))

            # Train agent
            agent = DynProg(model)
            _, pol1 = agent.train_val_iter()
            # Evaluate Policy
            total_reward1 = evaluate_policy(env_name, pol1, agent)
            cumul_rew_states1.append(total_reward1)

            #_, pol2 = agent.train_pol_iter()
            #total_reward2 = evaluate_policy(env_name, pol2, agent)
            #cumul_rew_states2.append(total_reward2)

        cumul_rew_list1.append(cumul_rew_states1)
        #cumul_rew_list2.append(cumul_rew_states2)

    # Value iteration plot
    plot_results("value_iteration", env_name, cumul_rew_list1, n_states, n_acts)

    # Policy iteration plot
    #plot_results("policy_iteration", env_name, cumul_rew_list2, n_states, n_acts)


def compare_rewards_random(env_name, n_states, n_acts):
    """
    Evaluates a random policy on different numbers of states and actions
    :param env_name: Name of the environment
    :param n_states: List with different numbers of states
    :param n_acts: List with different numbers of actions
    :return: List of cumulative rewards for each discretization
    """
    cumul_rew_list1 = []

    for na in n_acts:
        print("Number of discrete actions: ", na)
        cumul_rew_states1 = []
        for ns in n_states:
            print("Number of discrete states: ", ns)

            # Init. random policy
            arg = np.sqrt(ns) * 1j
            sts = np.mgrid[-np.pi:np.pi:arg, -8.0:8.0:arg].reshape(2, -1).T
            acts = list(np.linspace(-2, 2, na + 1))
            acts.remove(0)
            pol = [sts, np.random.choice(acts, sts.shape[0])]

            # Evaluate Policy
            total_reward = evaluate_policy(env_name, pol)
            cumul_rew_states1.append(total_reward)

        cumul_rew_list1.append(cumul_rew_states1)

    return cumul_rew_list1


random.seed(42)

env_names = ["Pendulum-v2", "Qube-v0"]
start = timer()
for env_name in env_names:
    sd = [3600, 4900] if env_name == "Pendulum-v2" else [2500, 3600, 4900]
    ad = [4]

    pickle.dump(sd, open("discretizations/sd_"+env_name+".arr", 'wb'))
    pickle.dump(ad, open("discretizations/ad_"+env_name+".arr", 'wb'))

    print("Random results:")
    rand_rews = compare_rewards_random(env_name, sd, ad)
    print("-----")
    compare_rewards(env_name, sd, ad)
print("Total time: ", timer() - start)

# TODO: Plot best results of Value function and Policy for Pendulum - v2
