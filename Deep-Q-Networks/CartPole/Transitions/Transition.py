class Transition:
    def __init__(self, obs, action, reward, next_obs, terminated):
        self.obs = obs
        self.action = action
        self.reward = reward
        self.next_obs = next_obs
        self.terminated = terminated