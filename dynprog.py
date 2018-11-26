
class ValIter:
    def __init__(self, policy, env, val):
        self.policy = policy
        self.env = env
        self.val = val

    def train(self):
        # Get state
        s = self.env.reset()
        return 0