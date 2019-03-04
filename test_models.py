import pickle

env_name = 'CartpoleStabShort-v0'
pol = pickle.load(open(env_name+".p", "rb"))

for _ in range(100):
    pol.rollout(render=True)
