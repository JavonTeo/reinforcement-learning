import gymnasium as gym
import numpy as np
import torch
import time
from datetime import datetime
from itertools import count
from torch.utils.tensorboard import SummaryWriter

from DQN import DQN
from ReplayMemory import PrioritizedReplayMemory
from Logger import Logger
from Transitions.Transition import Transition

class Agent:
    def __init__(self, env:gym.Wrapper, policy_net:DQN=None, target_net:DQN=None, replay_memory:PrioritizedReplayMemory=None, priority_scale=0.6, lr=0.0025, gamma=0.9, epsilon=1.0, epsilon_decay=1e-3, min_epsilon=0.01, device=torch.device('cpu')):
        self.env = env
        # obs is a array of 4 elements. There are also 4 possible actions.
        self.policy_net = policy_net if policy_net else DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.policy_net.to(device)
        self.target_net = target_net if target_net else DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.to(device)
        self.replay_memory = replay_memory if replay_memory else PrioritizedReplayMemory(10000)
        self.priority_scale = priority_scale
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.device = device
        self.criterion = self.loss_fn
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), self.lr, amsgrad=True)
        self.start_time = datetime.now()
        self.logdir_path = f"runs/{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        self.logger = Logger(self.logdir_path)
        self.logger.init_log(env, self, lr, gamma, epsilon, epsilon_decay, min_epsilon, device)
        self.writer = SummaryWriter(self.logdir_path)

    def loss_fn(self, pred_q_values, target_q_values, importance_weights):
        """
        Follows MSE loss, but with importance sampling weights.
        Taken from DQN with PER paper.
        """
        td_errors = target_q_values - pred_q_values
        importance_weighted_errors = importance_weights * td_errors
        loss = torch.mean(importance_weighted_errors ** 2)
        return loss

    def get_action(self, obs):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            self.policy_net.eval()
            with torch.no_grad():
                obs = torch.tensor(np.array(obs)).to(self.device)
                q_values = self.policy_net(obs)
            action = torch.argmax(q_values).item()
            return action
            
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon - self.epsilon_decay)

    def train(self, num_episodes, batch_size=128):
        beta_update = (1 - self.replay_memory.beta) / num_episodes
        for ep in range(1, num_episodes + 1):
            obs, info = self.env.reset()
            terminated = False
            episode_reward = 0

            for t in count(start=1):
                action = self.get_action(obs)
                self.policy_net.train()
                next_obs, reward, terminated, truncated, info =  self.env.step(action)
                episode_reward += reward
                transition = Transition(obs, action, reward, next_obs, int(terminated))
                self.replay_memory.push(transition)
                obs = next_obs
                # sample_probs = self.replay_memory.get_probs(self.priority_scale)
                mini_batch, importance_weights, sample_indices = self.replay_memory.sample(
                    batch_size, self.priority_scale, beta_update)
                # importance_weights = self.replay_memory.get_importance_weights(sample_probs, beta_update)
                if mini_batch is not None:
                    self.train_model(mini_batch, importance_weights, sample_indices, t)
                self.decay_epsilon()
                if terminated or truncated:
                    break
            
            self.logger.log_reward(ep, num_episodes, episode_reward, t)
            self.writer.add_scalar('Reward/train', episode_reward, ep)
 
        torch.save(self.policy_net.state_dict(), f'{self.logdir_path}/policy_{num_episodes}.pt')
        print(f"Model saved as policy_{num_episodes}.pt")

    def train_model(self, mini_batch, importance_weights, sample_indices, timestep):
        obss, actions, rewards, next_obss, terminateds = mini_batch
        obss = obss.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obss = next_obss.to(self.device)
        terminateds = terminateds.to(self.device)
        importance_weights = importance_weights.to(self.device)

        pred_q_values_all = self.policy_net(obss)
        # Get the Q-values of the actions taken
        pred_q_values = torch.gather(pred_q_values_all, 1, actions.unsqueeze(1)).squeeze()

        target_q_values_next_states = self.target_net(next_obss)
        target_q_values = rewards + self.gamma * torch.max(target_q_values_next_states, dim=1)[0] * (1 - terminateds)

        """
        Update priorities in replay memory
        """
        self.replay_memory.update_priorities(pred_q_values, target_q_values, sample_indices)

        loss = self.criterion(pred_q_values, target_q_values, importance_weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        N = 10
        if timestep % N == 0:
            policy_net_state_dict = self.policy_net.state_dict()
            target_net_state_dict = self.target_net.state_dict()
            # TAU = 0.005
            # for key in policy_net_state_dict:
            #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            # self.target_net.load_state_dict(target_net_state_dict)
            self.target_net.load_state_dict(policy_net_state_dict)

    def validate(self, num_episodes, val_env:gym.Wrapper, state_dict_path=None):
        episode_rewards = []
        if state_dict_path:
            state_dict = torch.load(state_dict_path, map_location=self.device, weights_only=True)
            self.policy_net.load_state_dict(state_dict)
        self.epsilon = self.min_epsilon
        for ep in range(1, num_episodes+1):
            obs, info = val_env.reset()
            terminated = False
            episode_reward = 0
            
            for t in count(start=1):
                val_env.render()
                with torch.no_grad():
                    action = self.get_action(obs)
                    next_obs, reward, terminated, truncated, info = val_env.step(action)
                    episode_reward += reward
                    obs = next_obs
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            self.logger.log_reward(ep, num_episodes, episode_reward, t, is_train=False)
            self.writer.add_scalar('Reward/test', episode_reward, ep)

        self.logger.log_val_ave_rewards(episode_rewards, ep)
        
    def close(self):
        self.env.close()
        end_time = datetime.now()
        self.logger.close(end_time, self.start_time)

    def __repr__(self):
        return "DQN Agent with Prioritized Replay Memory"