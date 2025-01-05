from collections import deque
import random
import numpy as np
import torch

class ReplayMemory:
    def __init__(self, capacity):
        self.replay_buffer = deque(maxlen = capacity)

    def push(self, transition):
        """ Save transition to replay buffer"""
        self.replay_buffer.append(transition)
    
    def sample(self, batch_size):
        """ Samples a random batch of transitions """
        if len(self) < batch_size:
            return None
        else:
            samp = random.sample(self.replay_buffer, batch_size)
            obss, actions, rewards, next_obss, terminateds = zip(*samp)
            return (torch.tensor(np.array(obss)).float(),
                    torch.tensor(np.array(actions)).long(),
                    torch.tensor(np.array(rewards)).float(),
                    torch.tensor(np.array(next_obss)).float(),
                    torch.tensor(np.array(terminateds)))
    
    def __len__(self):
        return len(self.replay_buffer)
