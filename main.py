import gymnasium as gym
import numpy as np

env = gym.make("LunarLander-v3", render_mode="human")
observation, info = env.reset()

# Initialize tracked variables
episode_over = False
reward_obtained = 0
value_fn = np.zeros((11, 11), dtype=np.float32)
# state is 8d vector: x-coord, y-coord, x-velocity, y-velocity, angle, angular velocity, bool left leg in contact w ground, bool right leg

def round_to_half(val):
    if val <= -2.5 and val < -2.25:
        return -2.5
    if val <= -2.25 and val < -1.75:
        return -2.0
    if val <= -1.75 and val < -1.25:
        return -1.5
    if val <= -1.25 and val < -0.75:
        return -1.0
    if val <= -0.75 and val < -0.25:
        return -0.5
    if val <= -0.25 and val < 0.25:
        return 0
    if val <= 0.25 and val < 0.75:
        return 0.5
    if val <= 0.75 and val < 1.25:
        return 1.0
    if val <= 1.25 and val < 1.75:
        return 1.5
    if val <= 1.75 and val < 2.25:
        return 2.0
    if val <= 2.25 and val <= 2.5:
        return 2.5
    
def arr_index(val):
    return int((val-(-2.5)) / (5/10))

def bellman_eqn(value_fn, reward, row, col):
    gamma = 0.9 # discount factor
    new_state_val = reward + gamma * 


while not episode_over:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    x_coord, y_coord = observation[0], observation[1]
    i, j = arr_index(round_to_half(x_coord)), arr_index(round_to_half(y_coord))
    value_fn[i:j] += 1

    # Value iteration
    delta = 0
    epsilon = 0
    while delta > epsilon:
        value_fn = value_fn
        delta = 0
        for row in value_fn:
            for col in value_fn:
                state_val = value_fn[row, col]
                new_state_val = bellman_eqn(value_fn, reward, row, col)
                abs_diff = abs(new_state_val - state_val)
                if abs_diff > delta:
                    delta = abs_diff

    episode_over = terminated or truncated

print(f"Total reward obtained:{reward_obtained}")
print(value_fn)
env.close()