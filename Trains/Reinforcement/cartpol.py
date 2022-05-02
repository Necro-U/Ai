import numpy as np 
import tflearn
import random
import gym
from tflearn.layers.core import input_data , dropout , fully_connected
from tflearn.layers.estimator import regression
from statistics  import mean,median
from collections import Counter

Lr=1e-3
import gym
env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()