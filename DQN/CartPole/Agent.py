import gymnasium as gym
import numpy as np
import torch

from DQN import DQN
from ReplayMemory import ReplayMemory

class Agent:
    def __init__(self, env:gym.Wrapper, policy_net:DQN=None, target_net:DQN=None, replay_memory:ReplayMemory=None, lr=0.01, gamma=0.9, epsilon=1.0, epsilon_decay=1e-3, min_epsilon=0.01):
        self.env = env
        obs, info = self.env.reset()
        self.policy_net = policy_net if policy_net else DQN(len(obs), self.env.action_space.n)
        self.target_net = target_net if target_net else DQN(len(obs), self.env.action_space.n)
        self.replay_memory = replay_memory if replay_memory else ReplayMemory(100)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), self.lr, amsgrad=True)

    def get_action(self, obs, is_val=False):
        if np.random.rand() <= self.epsilon and not is_val:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                obs = torch.tensor(np.array(obs))
                if is_val:
                    action = np.argmax(self.target_net(obs)[0]).item()
                else:
                    action = np.argmax(self.policy_net(obs)[0]).item()
                return action
            
    def decay_epsilon(self):
        self.epsilon -= self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon

    def train(self, num_episodes):
        for ep in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            terminated = False
            timestep = 0

            while not terminated:
                action = self.get_action(obs)
                # breakpoint()
                # if isinstance(action, torch.Tensor):
                #     breakpoint()
                timestep += 1
                next_obs, reward, terminated, truncated, info =  self.env.step(action)
                self.replay_memory.push((obs, action, reward, next_obs, terminated))
                obs = next_obs
                batch_size = 5
                mini_batch = self.replay_memory.sample(batch_size)
                if mini_batch is not None:
                    self.train_model(mini_batch, timestep)
                self.decay_epsilon()

    def train_model(self, mini_batch, timestep):
        obss, actions, rewards, next_obss, terminateds = mini_batch

        pred_q_values_all = self.policy_net(obss)
        pred_q_values = torch.gather(pred_q_values_all, 1, actions.unsqueeze(1)).squeeze()

        target_q_values_next_states = self.target_net(next_obss)
        target_q_values = rewards + self.gamma * torch.max(target_q_values_next_states, dim=1)[0]

        """
        Temporal Difference: target_q_values - pred_q_values
        """
        loss = self.criterion(pred_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Display some metrics
        

        N = 10
        if timestep % N == 0:
            print(loss.item())
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def validate(self, num_episodes, val_env:gym.Wrapper):
        ep_rewards = []
        for ep in range(num_episodes):
            obs, info = val_env.reset()
            terminated = False
            ep_reward = 0
            
            while not terminated:
                val_env.render()
                with torch.no_grad():
                    action = self.get_action(obs, True)
                    next_obs, reward, terminated, truncated, info = val_env.step(action)
                    ep_reward += reward
                    obs = next_obs
            
            ep_rewards.append(ep_reward)
        
        print(ep_rewards)