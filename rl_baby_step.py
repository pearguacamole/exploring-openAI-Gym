import gymnasium as gym
import random

env = gym.make("CartPole-v1", render_mode = 'human')

episodes =10
for episode in range(1,episodes+1):
    state = env.reset()
    done = False
    donne = False
    score =0
    while not (done or donne):
        action = random.choice([0,1])
        _,reward,done,donne,_=env.step(action)
        score+=reward
        #env.render()

    print(f"Episode {episode}, {score=}")
env.close()