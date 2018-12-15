import gym
import numpy as np

from build_model import load_model
from fun_approximator import FitNN
import quanser_robots
import matplotlib.pyplot as plt


# Plot samples
def pick_color(rew):
    if rew<-10:
        return "red"
    elif rew>=-10 and rew<-1:
        return "yellow"
    else:
        return "green"

env_name = "Pendulum-v2"
env = gym.make(env_name)
s_dim = env.reset().shape[0]
net = FitNN(s_dim+1, s_dim, env, False)
points = net.rollout(3000)

states = [p[0] for p in points]
actions = [p[2][0] for p in points]

plt.scatter([p[0][0] for p in points], [p[1][1] for p in points], c=[pick_color(p[3]) for p in points])
plt.show()

model = load_model(env_name, (3000, 4))
c = []
for s,a in zip(range(model.states.shape[0]), actions):
    c.append(pick_color(-(model.reward_matrix[s][model.actions.index(a)]**-1)))
plt.scatter([s[0] for s in model.states], [s[1] for s in model.states], c=c)
plt.show()
