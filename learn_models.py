""" Main file for testing the implementation for the challenge."""
import random

import gym
import quanser_robots
import dill
import pickle
import torch
import numpy as np
from build_model import ModelBuilder
from fun_approximator import FitNN
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def pick_color(rew):
    """
    Pick a color based on a reward
    :param rew: Reward value
    :return: Color string
    """
    if rew < -10:
        return "red"
    elif rew >= -10 and rew < -1:
        return "yellow"
    else:
        return "green"


def plot_samples(points):
    """
    Plot sampled states
    :param points: Samples
    """
    plt.scatter([p[0][0] for p in points], [p[1][1] for p in points], c=[pick_color(p[3]) for p in points])
    plt.title("Sampled states together with rewards")
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.show()


def plot_model(model, actions):
    """
    Plots learnt model
    :param model: A ModelBuilder object
    """
    c = []
    for s, a in zip(range(model.states.shape[0]), actions):
        c.append(pick_color(-(model.reward_matrix[s][(np.abs(model.actions - a)).argmin()] ** -1)))
    plt.scatter([s[0] for s in model.states], [s[1] for s in model.states], c=c)
    plt.title("Learnt model: Sampled states together with rewards")
    plt.xlabel("Angle")
    plt.ylabel("Velocity")
    plt.show()


random.seed(42)

env_names = ["Pendulum-v2", "Qube-v0"]
sample_num = [(3000, 9000), (3000, 22000)]

start = timer()
for env_name, n_samps in zip(env_names, sample_num):
    print(env_name)
    env = gym.make(env_name)
    """
    # Dimension of states
    s_dim = env.reset().shape[0]
    reward = FitNN(s_dim + 1, 1, env, False)
    dynamics1 = FitNN(s_dim + 1, s_dim, env, True)
    dynamics2 = FitNN(s_dim + 1, s_dim, env, True, True)
    
    # Sample training data
    print("Rollout policy...")
    points = reward.rollout(n_samps[0])  # See plots in figures dir for optimal number of samples

    # Learn dynamics and rewards
    reward.learn(points, 256, 64)

    points_left = dynamics1.rollout(n_samps[1] * 2)
    points_right = dynamics2.rollout(n_samps[1] * 2)

    dynamics1.learn(points_left, 1024, 64)
    dynamics2.learn(points_right, 1024, 64)

    o1, o2 = 0, 0
    for pl, pr in zip(points_left, points_right):
        predl = dynamics1.predict(torch.Tensor([pl[0][0], pl[0][1], pl[2][0]]))
        predr = dynamics2.predict(torch.Tensor([pr[0][0], pr[0][1], pr[2][0]]))
        reall = pl[1]
        realr = pr[1]
        if np.sum(np.abs(predl - reall)) > 1:
            o1 += 1
        if np.sum(np.abs(predr - realr)) > 1:
            o2 += 1
    print(o1)
    print(o2)

    # Save for later use
    # TODO: Error when trying to save NN for Qube. Why?
    pickle.dump(reward, open("nets/rew_"+env_name+".fitnn", 'wb'))
    pickle.dump(dynamics1, open("nets/dyn1_"+env_name+".fitnn", 'wb'))
    pickle.dump(dynamics2, open("nets/dyn2_"+env_name+".fitnn", 'wb'))
    """
    reward = pickle.load(open("nets/rew_" + env_name + ".fitnn", 'rb'))
    dynamics1 = pickle.load(open("nets/dyn1_" + env_name + ".fitnn", "rb"))
    dynamics2 = pickle.load(open("nets/dyn2_" + env_name + ".fitnn", "rb"))

    sd = pickle.load(open("discretizations/sd_" + env_name + ".arr", 'rb'))
    ad = pickle.load(open("discretizations/ad_" + env_name + ".arr", "rb"))

    for s in sd:
        for a in ad:
            # Create points for most important states
            mid_points = []
            acts = list(np.linspace(-2, 2, a + 1))
            acts.remove(0)
            arg = np.sqrt(int(s / 100)) * 1j
            sts = np.mgrid[-0.2:0.2:arg, -0.4:0.4:arg].reshape(2, -1).T
            for state in sts:
                for action in acts:
                    inp = torch.Tensor([state[0], state[1], action])
                    nxt_state = dynamics1.predict(inp) if state[0] < 0 else dynamics2.predict(inp)
                    rew = reward.predict(inp)[0]
                    mid_points.append([state, nxt_state, np.array([action]), rew])
            """
            # Create points for other states
            other_points = []
            arg = np.sqrt(s) * 1j
            sts = np.mgrid[-np.pi:np.pi:arg, -8:8:arg].reshape(2, -1).T
            for state in sts:
                for action in acts:
                    inp = torch.Tensor([state[0], state[1], action])
                    nxt_state = dynamics1.predict(inp) if state[0] < 0 else dynamics2.predict(inp)
                    rew = reward.predict(inp)[0]
                    other_points.append([state, nxt_state, np.array([action]), rew])
            """
            other_points = reward.rollout(25000)
            points = mid_points + other_points
            # plot_samples(points)
            actions = [p[2][0] for p in points]

            print("Building model:")
            print(str(s) + " states")
            print(str(a) + " actions")
            model = ModelBuilder(env_name, reward, dynamics1, dynamics2)
            model.build_model(points, (s + int(s / 100), a))
            model.save_model()

            # Plot model states with rewards
            #plot_model(model, actions)

    print(env_name + " done")
print("Total time: ", timer() - start)
