""" Compares model accuracy for learning dynamics and rewards on different numbers of samples"""


import gym
import quanser_robots
import matplotlib.pyplot as plt

from fun_approximator import FitNN

def compare_models(env_name, max_samples):
    """
    Trains NNs on rewards and dynamics for different numbers of samples and plots results
    :param env_name: Name of the current environment
    :param max_samples: Max. number of samples for learning
    """
    print(env_name)

    env = gym.make(env_name)
    s_dim = env.reset().shape[0]

    loss_dyn, loss_rew = [], []

    ns = range(1000, max_samples, 1000)
    for n in ns:
        print("Number of samples: ", n)

        # Init. NNs
        reward = FitNN(s_dim + 1, 1)
        dynamics = FitNN(s_dim + 1, s_dim)

        # Generate training data
        points = reward.rollout(n, env)

        # Learn and append loss
        reward.learn(False, points)
        loss_rew.append(reward.total_loss)

        dynamics.learn(True, points)
        loss_dyn.append(dynamics.total_loss)

    # Make it pretty!
    plt.plot(ns, loss_rew, label="Reward learning")
    plt.plot(ns, loss_dyn, label="Dynamics learning")
    plt.title("Comparison of learning for different number of samples")
    plt.xlabel("Number of samples")
    plt.ylabel("Total loss after training")
    plt.legend()

    # Save plot
    fig = plt.gcf()
    fig.savefig('figures/model_comparison_'+env_name+'.png')

    # Show plot
    plt.show()

# Pendulum
compare_models("Pendulum-v2", 10000)

# Qube
#compare_models("Qube-v0", 3000)