import gymnasium as gym
import torch

from Agent import Agent

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent = Agent(env)
    agent.train(1000)

    val_env = gym.make("CartPole-v1", render_mode="human")
    agent.validate(10, val_env)
    val_env.close()