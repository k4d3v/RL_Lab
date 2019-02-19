import gym
import quanser_robots
from timeit import default_timer as timer
import pickle

import evaluate

def test(policy, env_name):
    start = timer()

    # Evaluate Model after learning with 100 rollouts
    eval = evaluate.Evaluator(policy, gym.make(env_name))
    ev = eval.evaluate(2, True)

    end = timer()
    print("Done evaluating learnt policy, ", end - start)
    print("Total reward:", ev)


#env_names = ['CartpoleStabShort-v0', 'CartpoleStabLong-v0', 'CartpoleSwingShort-v0', 'CartpoleSwingLong-v0', 'BallBalancerSim-v0']
env_names = ['CartpoleSwingShort-v0', 'BallBalancerSim-v0']

for env_name in env_names:
    print(env_name)

    iters = [40]  # Number of iterations of training
    traj_samps = 80 # Number of sampled trajs per iteration during training

    for i in iters:
        policy = pickle.load(open("policies/"+env_name+"_"+str(i)+"_"+str(traj_samps)+".slp", "rb" ))
        test(policy, env_name)
