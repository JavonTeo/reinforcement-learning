import torch.nn as nn

class DQN(nn.Sequential):
    """ Deep Q-Network implementation 
    For a state, it outputs the expected return for taking each action in that state (or Q-value).
    """
    def __init__(self, n_observations, n_actions):
        super().__init__(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        