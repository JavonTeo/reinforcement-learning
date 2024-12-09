from collections import defaultdict
import gymnasium as gym
import numpy as np
import os
import pickle

save_directory = "./q_values_checkpoints/"

class Agent:
    """
    Discrete actions, Discrete states.
    Uses q table to store q values.
    """
    def __init__(self, env: gym.wrappers.RecordEpisodeStatistics, lr=0.01, gamma=0.9, initial_epsilon=1.0, epsilon_decay=0.01, final_epsilon=0.1, q_values=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        if q_values:
            self.q_values  = defaultdict(lambda: np.zeros(env.action_space.n), q_values)
        else:
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) # for each empty obs

        self.training_error = []

    def get_action(self, obs):
        if np.random.random() <= self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs])) # obs is the observation received at a particular point of time
        
    def update(self, obs, action, reward, terminated, next_obs):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = reward + self.gamma * future_q_value - self.q_values[obs][action]
        self.q_values[obs][action] += self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    def report(self):
        return self.env.episode_returns
    
    def save_q_values(self, episode):
        if episode % 1000 == 0:
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, f"q_table_ep_{episode}.pkl")
            q_values = dict(self.q_values)
            with open(save_path, "wb") as f:
                pickle.dump(q_values, f)
            print(f"Saved Q-table at episode {episode} to {save_path}")
            print(f"Episode {episode}: {self.report()} reward")


def train():
    n_episodes = 100_000
    epsilon_decay = 1.0 / (n_episodes / 2)

    env = gym.make("Blackjack-v1", sab=False, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = Agent(env=env, epsilon_decay=epsilon_decay)

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        # agent.save_q_values(episode)

def test():
    n_episodes_collected = 1000
    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1000)
    with open("q_values_checkpoints/q_table_ep_21000.pkl", "rb") as f:
        q_table = pickle.load(f)
    trained_agent = Agent(env=env, initial_epsilon=0, final_epsilon=0, q_values=q_table)
    new_agent = Agent(env=env)
    agent = trained_agent
    total_reward = [0, 0, 0, 0, 0]
    for i in range(5):
        ep = 0
        while ep < n_episodes_collected:
            obs, info = env.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                total_reward[i] += reward

                done = terminated or truncated
                obs = next_obs

            ep += 1
        
        total_reward[i] /= n_episodes_collected
        print(total_reward[i])

    print(total_reward)

test()