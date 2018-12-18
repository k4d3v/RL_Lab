"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""
import numpy as np
import torch

from build_model import ModelBuilder, load_model
from dyn_prog import DynProg
from fun_approximator import FitNN

info = dict(
    group_number=None,  # change if you are an existing seminar/project group
    authors="Leon Keller; Xiao Han; Zlatko Kolev",
    description="Trains NNs on reward and dynamics. For dynamics, two NNs are trained, depending on the angle. Does value iteration.")


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """
    env_name = "Pendulum-v2"
    # Number of states and actions
    s = 4900
    a = 4

    # Dimension of states
    s_dim = env.reset().shape[0]
    reward = FitNN(s_dim + 1, 1, env, False)
    dynamics1 = FitNN(s_dim + 1, s_dim, env, True)
    dynamics2 = FitNN(s_dim + 1, s_dim, env, True, True)

    # Sample training data
    print("Rollout policy...")
    points = reward.rollout(0.2*max_num_samples)  # See plots in figures dir for optimal number of samples

    # Learn dynamics and rewards
    reward.learn(points, 256, 64)

    points_left = dynamics1.rollout(max_num_samples)
    points_right = dynamics2.rollout(max_num_samples)

    dynamics1.learn(points_left, 512, 64)
    dynamics2.learn(points_right, 512, 64)

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

    other_points = reward.rollout(max_num_samples)
    points = mid_points + other_points

    print("Building model:")
    print(str(s) + " states")
    print(str(a) + " actions")
    model = ModelBuilder(env_name, reward, dynamics1, dynamics2)
    model.build_model(points, (s + int(s / 100), a))
    model.save_model()

    return lambda obs, act: (dynamics1.predict(torch.Tensor([obs[0], obs[1], act])) if obs[0]<0
                             else dynamics2.predict(torch.Tensor([obs[0], obs[1], act])),
                             reward.predict(torch.Tensor([obs[0], obs[1], act])))


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    env_name = "Pendulum-v2"
    s = 4900
    a = 4
    the_model = load_model(env_name, (s+int(s/100), a))

    agent = DynProg(the_model)
    _, pol1 = agent.train_val_iter()

    return lambda obs: pol1[((model.states - obs) ** 2).sum(1).argmin()]
