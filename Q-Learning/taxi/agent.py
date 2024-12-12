from collections import defaultdict
import os
import pickle
import gymnasium as gym
import numpy as np


class Agent:
    def __init__(self, env:gym.Wrapper, lr=0.01, gamma=0.9, initial_epsilon=1.0, epsilon_decay=0.01, final_epsilon=0.1, checkpoint=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        if checkpoint:
            with open(checkpoint, "rb") as f:
                q_table = pickle.load(f)
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n), q_table)
        else:
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) # for each empty obs

        self.training_error = []

    def get_action(self, obs):
        if np.random.rand() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))
        
    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.gamma * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_q_values(self, episode, step, save_dir):
        """
        episode: the current episode
        step: step size for saving the q-value
        """
        if episode % step == 0:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"q_table_ep_{episode}.pkl")
            q_values = dict(self.q_values)
            with open(save_path, "wb") as f:
                pickle.dump(q_values, f)
            print(f"Saved Q-table at episode {episode} to {save_path}")


def train(n_episodes, step, checkpoint=None):
    """
    n_episodes: number of episodes to train
    """
    save_directory = './q_values_checkpoints/'

    env = gym.make("Taxi-v3")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200) # set truncated episode limit
    agent = Agent(env, checkpoint=checkpoint)

    for ep in range(1, n_episodes + 1):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        agent.save_q_values(ep, step, save_directory)
        if ep % step == 0 :
            print(f"Episode {ep}: {reward} reward")


def test(cp_pkl):
    """
    pkl_fname: name of pickle file containing q_values to load
    """
    n_episodes_collected = 1000
    env = gym.make("Taxi-v3")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes_collected)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200) # set truncated episode limit
    trained_agent = Agent(env=env, initial_epsilon=0, final_epsilon=0, checkpoint=cp_pkl)
    agent = trained_agent
    reward_record = [0, 0, 0, 0, 0]
    for i in range(len(reward_record)):
        ep = 0
        while ep < n_episodes_collected:
            obs, info = env.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                reward_record[i] += reward

                done = terminated or truncated
                obs = next_obs

            ep += 1
        
        reward_record[i] /= n_episodes_collected

    print(f"Reward obtained for {len(reward_record)} iterations over {n_episodes_collected} episodes each: {reward_record}")

if __name__ == "__main__":
    pkl_fname = "q_values_checkpoints/q_table_ep_40000.pkl"
    # train(500000, step=100000, checkpoint=pkl_fname)
    test(pkl_fname)