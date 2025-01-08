import gymnasium as gym
import logging
import os

class Logger:
    def __init__(self, dirPath):
        self.logger = logging.getLogger(__name__)
        os.makedirs(dirPath, exist_ok=True)
        logging.basicConfig(filename=f'{dirPath}/log.log', level=logging.INFO)
    
    def init_log(self, env:gym.Wrapper, agent, lr, gamma, epsilon, epsilon_decay, min_epsilon, device):
        self.logger.info(f"Environment: {env.unwrapped.spec.id}")
        self.logger.info(f"Agent: {agent}")
        self.logger.info(f"Learning rate: {lr}")
        self.logger.info(f"Gamma: {gamma}")
        self.logger.info(f"Epsilon: {epsilon}")
        self.logger.info(f"Epsilon decay: {epsilon_decay}")
        self.logger.info(f"Minimum epsilon: {min_epsilon}")
        self.logger.info(f"Device used: {device}")
    
    def log_reward(self, ep, total_eps, reward, timestep, is_train=True):
        if is_train:
            self.logger.info(f"[TRAIN][ep {ep}/{total_eps}][TIMESTEP {timestep}]: {reward}")
        else:
            self.logger.info(f"[VALIDATION][ep {ep}/{total_eps}][TIMESTEP {timestep}]: {reward}")
    
    def log_val_ave_rewards(self, rewards, total_eps):
        self.logger.info(f"Validation average rewards over {total_eps} eps: {sum(rewards)/total_eps}")

    def close(self, end_time, start_time):
        self.logger.info(f"Training completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total duration: {end_time - start_time}")