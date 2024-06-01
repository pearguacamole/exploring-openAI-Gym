import gymnasium as gym
import random
import tensorflow
#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam


class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        return obs, reward, done, {}



#this solves "ImportError: cannot import name '__version__' from 'tensorflow.keras'" from rl.agents import
from keras import __version__
tensorflow.keras.__version__ = __version__


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

#env = gym.make("CartPole-v1", render_mode = 'human')
#env = gym.make("CartPole-v1")
env = GymWrapper(gym.make("CartPole-v1"))


states = env.observation_space.shape[0]
actions = env.action_space.n


model = Sequential()
model.add(Flatten(input_shape=(1, states))) 
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000,window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup = 10,
    target_model_update = 0.01
)

agent.compile(Adam(lr=0.001),metrics=['mae'])
agent.fit(env,nb_steps=10000,visualize=False,verbose=2)
env = GymWrapper(gym.make("CartPole-v1",render_mode='human'))
results = agent.test(env,nb_episodes=10,visualize=True)
print(results.history['episode_reward'])

# episodes =10
# for episode in range(1,episodes+1):
#     state = env.reset()
#     done = False
#     donne = False
#     score =0
#     while not (done or donne):
#         action = random.choice([0,1])
#         _,reward,done,donne,_=env.step(action)
#         score+=reward
#         #env.render()

#     print(f"Episode {episode}, {score=}")
env.close()