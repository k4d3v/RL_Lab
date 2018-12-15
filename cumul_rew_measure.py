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


def pick_color(rew):
    if rew<-10:
        return "red"
    elif rew>=-10 and rew<-1:
        return "yellow"
    else:
        return "green"

def pick_color_act(act):
    if act==2:
        return "red"
    elif act==1:
        return "orange"
    elif act == -2:
        return "blue"
    elif act==-1:
        return "cyan"

def plot_samples(points):
    # Plot samples
    plt.scatter([p[0][0] for p in points], [p[1][1] for p in points], c=[pick_color(p[3]) for p in points])
    plt.show()

def plot_results(algo_name, env_name, cumul_rew_list, n_states, n_acts):
    """
    Plots cumulative rewards for a given algo for different state and action space discretizations
    :param algo_name: Name of the algorithm
    :param env_name: Name of the environment
    :param cumul_rew_list:
    :param n_states: List with different numbers of states
    :param n_acts: List with different numbers of actions
    """
    """
    # Plot random results
    global rand_rews
    for ai, i in zip(n_acts, range(len(n_acts))):
        plt.plot(n_states, rand_rews[i], label="Baseline of random policy, "+str(ai)+" actions")
    """
    # Iterate over number of actions and plot curves with values for different state numbers
    for na in range(len(cumul_rew_list)):
        plt.plot(n_states, cumul_rew_list[na], label=str(n_acts[na]) + " actions")
    plt.title("Comparison of " + algo_name + " results for different discretizations on "+env_name)
    plt.xlabel("Number of discrete entries in the state space")
    plt.ylabel("Cumulative reward after training")
    plt.legend()

    # Save plot
    fig = plt.gcf()
    fig.savefig('figures/cumul_rew_' + algo_name + '_comparison_' + env_name + '.png')

    # Show plot
    plt.show()


def plot_pol(states, actions):
    # Plot samples
    plt.scatter([s[0] for s in states], [s[1] for s in states], c=[pick_color_act(act) for act in actions])
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

    plot_pol(states, actions)

    total_reward = 0.0
    for i in range(100):
        #print("Episode: ", i)

        #plot_pol(states, actions)

        new_samples, new_samples_left, new_samples_right = [], [], []

        # Reset the environment
        observation = val_env.reset()
        done = False

        episode_reward = 0.0
        while not done:
            #old_state = observation

            idx = ((states - observation) ** 2).sum(1).argmin()
            action = [actions[idx]]

            #pred_obs = agent.model.dynamics1.predict(torch.Tensor([states[idx][0], states[idx][1], action[0]])) if states[idx][0]<0 else agent.model.dynamics2.predict(torch.Tensor([states[idx][0], states[idx][1], action[0]]))

            observation, rew, done, _ = val_env.step(action)  # Take action
            """
            new_samples.append([old_state, observation, action, rew])
            if old_state[0]<0:
                new_samples_left.append([old_state, observation, action, rew])
            else:
                new_samples_right.append([old_state, observation, action, rew])
            """
            episode_reward += rew
        """
        if agent is not None:
            # Fit reward and dynamics onto new points
            agent.model.reward.learn(new_samples, 256, 64)
            agent.model.dynamics1.learn(new_samples_left, 512, 64)
            agent.model.dynamics2.learn(new_samples_right, 512, 64)
            agent.model.refresh(new_samples)
            plot_samples(new_samples)

            # Learn again
            _, pol = agent.train_val_iter()
            states, actions = pol
        """

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
            model = load_model(env_name, (ns, na))

            # Train agent
            agent = DynProg(model)
            _, pol1 = agent.train_val_iter()
            # Evaluate Policy
            total_reward1 = evaluate_policy(env_name, pol1, agent)
            cumul_rew_states1.append(total_reward1)

            #_, pol2 = agent.train_pol_iter()
            #total_reward2 = evaluate_policy(env_name, pol2)
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
            acts = np.linspace(-2, 2, na)
            pol = [sts, np.random.choice(acts, sts.shape[0])]

            # Evaluate Policy
            total_reward = evaluate_policy(env_name, pol)
            cumul_rew_states1.append(total_reward)

        cumul_rew_list1.append(cumul_rew_states1)

    return cumul_rew_list1


random.seed(42)

### Pendulum
sd = range(2000, 7000, 1000)
ad = [4]

pickle.dump(sd, open("discretizations/sd_Pendulum-v2.arr", 'wb'))
pickle.dump(ad, open("discretizations/ad_Pendulum-v2.arr", 'wb'))

#print("Random results:")
#rand_rews = compare_rewards_random("Pendulum-v2", sd, ad)
#print("-----")
start = timer()
compare_rewards("Pendulum-v2", sd, ad)
print("Total time: ", timer()-start)

