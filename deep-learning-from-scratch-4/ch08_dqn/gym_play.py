import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v0', render_mode="human")
state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, truncated, info = env.step(action)
env.close()
