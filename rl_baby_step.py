import gymnasium as gym
import random
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam

#wrapper to make gymansium compatable with keras-rl2
class GymWrapper(gym.Wrapper):  
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, trunc, info  = self.env.step(action)
        return obs, reward, done or trunc, {}                   #done or trunc handles time limit <ends the simulation after 500 steps>
#done defines if the task has failed, trunc maintains the time limits, stop the episode if any turns TRUE


#this solves "ImportError: cannot import name '__version__' from 'tensorflow.keras'" from rl.agents import
from keras import __version__
tensorflow.keras.__version__ = __version__


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
#normal definition for making environment 
#env = gym.make("CartPole-v1", render_mode = 'human')
#env = gym.make("CartPole-v1")

#new environment with wrapper
env = GymWrapper(gym.make("CartPole-v1"))

#number of states and actions in the environment agent can interact with and perform
states = env.observation_space.shape[0]
actions = env.action_space.n

#defining underlying neural net
model = Sequential()
model.add(Flatten(input_shape=(1, states))) 
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))


#defining agent with MLP as the underlying model
agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000,window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions = actions,
    nb_steps_warmup = 10,
    target_model_update = 0.01
)
#model compilation and training
agent.compile(Adam(lr=0.001),metrics=['mae']) #optimized for mean absolute error
agent.fit(env,nb_steps=100000,visualize=False,verbose=2)

#creating new environment with render mode as human to visualize the testing phase
#Will also need to make change to callback.py in /site-packages/rl/callbacks.py <path to keras-rl2>
#In callback.py inside Visualizer class in on_action_end function 
#Change   self.env.render(make='human') ---> self.env.render()
# This deals with API difference in gym and gymnasium
env = GymWrapper(gym.make("CartPole-v1",render_mode='human'))


results = agent.test(env,nb_episodes=10,visualize=True)
print(results.history['episode_reward'])



# Testing out initial environment with random moves
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