import gym

from fun_approximator import FitNN
import quanser_robots
import matplotlib.pyplot as plt

env_name = "Pendulum-v2"
env = gym.make(env_name)
s_dim = env.reset().shape[0]
dynamics = FitNN(s_dim+1, s_dim, env, True)
points = dynamics.rollout(2000)

states = [p[0] for p in points]

# Plot samples
plt.scatter([s[0] for s in states], [s[1] for s in states])
plt.show()

print("hi")