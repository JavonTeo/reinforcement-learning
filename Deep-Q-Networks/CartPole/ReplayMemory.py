from collections import deque
import random
import numpy as np
import torch

from Transitions.Transition import Transition
from Transitions.PrioritizedTransition import PrioritizedTransition

class ReplayMemory:
    def __init__(self, capacity):
        self.replay_buffer = deque(maxlen = capacity)

    def push(self, transition:Transition):
        """ Save transition to replay buffer"""
        self.replay_buffer.append(transition)
 
    def sample(self, batch_size):
        """ Samples a random batch of transitions """
        if len(self) < batch_size:
            return None
        else:
            samp = random.sample(self.replay_buffer, batch_size)
            obss, actions, rewards, next_obss, terminateds = (
                [transition.obs for transition in samp],
                [transition.action for transition in samp],
                [transition.reward for transition in samp],
                [transition.next_obs for transition in samp],
                [transition.terminated for transition in samp],
            )
            return (torch.tensor(np.array(obss)).float(),
                    torch.tensor(np.array(actions)).long(),
                    torch.tensor(np.array(rewards)).float(),
                    torch.tensor(np.array(next_obss)).float(),
                    torch.tensor(np.array(terminateds)))

    def __len__(self):
        return len(self.replay_buffer)


class PrioritizedReplayMemory:
    def __init__(self, capacity, priority_epsilon=1e-5, beta=0.4):
        """
        Note: Both the transitions and the priorities are stored in the replay buffer.
        """
        self.replay_buffer = deque(maxlen = capacity)
        self.priority_epsilon = priority_epsilon
        self.beta = beta

    def push(self, transition:Transition):
        """ Save transition to replay buffer"""
        priority = max([transition.priority for transition in self.replay_buffer], default=1)
        self.replay_buffer.append(PrioritizedTransition(transition, priority))

    def update_priorities(self, pred_q_values, target_q_values, sample_indices):
        td_errors = target_q_values - pred_q_values
        priorities = abs(td_errors) + self.priority_epsilon 
        for i, p in zip(sample_indices, priorities):
            self.replay_buffer[i].priority = p.item()

    def get_probs(self, priority_scale):
        scaled_priorities = np.array([transition.priority for transition in self.replay_buffer]) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance_weights(self, sample_probabilities, beta_update):
        self.beta = min(1, self.beta + beta_update)
        importance = np.power(len(self.replay_buffer) * sample_probabilities, -self.beta)
        importance_normalized = importance / max(importance)
        return torch.tensor(importance_normalized)
    
    def sample(self, batch_size, priority_scale, beta_update):
        """ Samples a random batch of transitions """
        if len(self) < batch_size:
            return None, None, None
        else:
            #if len(self) >= 200:
            #    breakpoint()
            sample_probabilities = self.get_probs(priority_scale)
            sample_indices = random.choices(range(len(self.replay_buffer)), weights=sample_probabilities, k=batch_size)
            samp = np.array(self.replay_buffer, dtype=object)[sample_indices]
            obss, actions, rewards, next_obss, terminateds = (
                [transition.obs for transition in samp],
                [transition.action for transition in samp],
                [transition.reward for transition in samp],
                [transition.next_obs for transition in samp],
                [transition.terminated for transition in samp],
            )
            importance_weights = self.get_importance_weights(sample_probabilities[sample_indices], beta_update)
            return (torch.tensor(np.array(obss)).float(),
                    torch.tensor(np.array(actions)).long(),
                    torch.tensor(np.array(rewards)).float(),
                    torch.tensor(np.array(next_obss)).float(),
                    torch.tensor(np.array(terminateds))), importance_weights, sample_indices

    def __len__(self):
        return len(self.replay_buffer)