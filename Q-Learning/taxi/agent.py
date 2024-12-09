from collections import defaultdict
import os
import pickle
import gymnasium as gym
import numpy as np

save_directory = './q_values_checkpoints/'

class Agent:
    def __init__(self, env:gym.Wrapper, lr=0.01, gamma=0.9, initial_epsilon=1.0, epsilon_decay=0.01, final_epsilon=0.1, q_values=None):
        self.env = env
        self.lr = lr
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = epsilon_decay

        if q_values:
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n), q_values)
        else:
            self.q_values = defaultdict(lambda: np.zeros(env.action_space.n)) # for each empty obs

        self.training_error = []
        # self.last_reward = 0

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
        
        # self.last_reward = reward

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    
    # def report(self):
        # return self.last_reward

    def save_q_values(self, episode):
        if episode % 500 == 0:
            os.makedirs(save_directory, exist_ok=True)
            save_path = os.path.join(save_directory, f"q_table_ep_{episode}.pkl")
            q_values = dict(self.q_values)
            with open(save_path, "wb") as f:
                pickle.dump(q_values, f)
            print(f"Saved Q-table at episode {episode} to {save_path}")
            # print(f"Episode {episode}: {self.report()} reward")


def train(n_episodes):
    """
    n_episodes: number of episodes to train
    """
    env = gym.make("Taxi-v3")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200) # set truncated episode limit
    agent = Agent(env)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()
        agent.save_q_values(ep)
        print(f"Episode {ep}: {reward} reward")


def test():
    n_episodes_collected = 5
    env = gym.make("Taxi-v3", render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes_collected)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200) # set truncated episode limit
    with open("1700_Dec8_2024_q_values_checkpoints/q_table_ep_29500.pkl", "rb") as f:
        q_table = pickle.load(f)
    trained_agent = Agent(env=env, initial_epsilon=0, final_epsilon=0, q_values=q_table)
    new_agent = Agent(env=env)
    agent = trained_agent
    total_reward = [0, 0, 0, 0, 0]
    for i in range(1):
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

if __name__ == "__main__":
    # train(30000)
    test()