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


class PrioritizedReplayMemory:
    def __init__(self, capacity, priority_epsilon=1e-5):
        self.replay_buffer = deque(maxlen = capacity)
        self.priorities = deque(maxlen = capacity)
        self.priority_epsilon = priority_epsilon

    def push(self, transition):
        """ Save transition to replay buffer"""
        self.replay_buffer.append(transition)
        self.priorities.append(max(self.priorities, default=1))

    def update_priorities(self, sample_indices, td_errors):
        priorities = abs(td_errors) + self.priority_epsilon 
        for i, p in zip(sample_indices, priorities):
            self.priorities[i] = p.item()

    def get_probs(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance_weights(self, sample_probabilities):
        importance = 1 / len(self.replay_buffer) * 1 / sample_probabilities
        importance_normalized = importance / max(importance)
        return torch.tensor(importance_normalized)
    
    def sample(self, batch_size, priority_scale):
        """ Samples a random batch of transitions """
        if len(self) < batch_size:
            return None, None, None
        else:
            sample_probabilities = self.get_probs(priority_scale)
            sample_indices = random.choices(range(len(self.replay_buffer)), weights=sample_probabilities, k=batch_size)
            samp = np.array(self.replay_buffer, dtype=object)[sample_indices]
            obss, actions, rewards, next_obss, terminateds = zip(*samp)
            importance_weights = self.get_importance_weights(sample_probabilities[sample_indices])
            return (torch.tensor(np.array(obss)).float(),
                    torch.tensor(np.array(actions)).long(),
                    torch.tensor(np.array(rewards)).float(),
                    torch.tensor(np.array(next_obss)).float(),
                    torch.tensor(np.array(terminateds))), importance_weights, sample_indices

    def __len__(self):
        return len(self.replay_buffer)