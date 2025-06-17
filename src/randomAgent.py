# No imports Needed

# ========= Random Agent ========= #
class RandomAgent:

    def __init__(self, env):
        self.env = env

    def act(self, state):
        return self.env.action_space.sample()
    
    def step(self, state, action, reward, next_state, done):
        pass # Random agent does not learn
