import gymnasium as gym

class Env():
    def __init__(self, task_name):
        try:
            self.env = gym.make(task_name)
        except gym.error.NameNotFound as e:
            print(f"Error in creating environment: {e}")