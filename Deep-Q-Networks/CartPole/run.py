import gymnasium as gym
import torch

from Agent import Agent

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    num_episodes = 1000

    agent = Agent(env, device=device)
    agent.train(num_episodes)

    val_env = gym.make("CartPole-v1", render_mode="human")
    agent.validate(10, val_env, f"policy_{num_episodes}.pt")
    val_env.close()