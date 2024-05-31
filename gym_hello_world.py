# Hello world for OpenAI gymnasium

import gymnasium as gym

#environment creation
env = gym.make("LunarLander-v2",render_mode='human')#render_mode = 'human' makes it so that the environment is rendered

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()#taking a sample action out of action space
    observation, reward, terminated, truncated, info = env.step(action) #environment resopnse according to that action
    print(f'{observation=}, {reward=}, {terminated=}, {truncated=}, {info=}')

    if terminated or truncated:
        observation, info = env.reset()
env.close()