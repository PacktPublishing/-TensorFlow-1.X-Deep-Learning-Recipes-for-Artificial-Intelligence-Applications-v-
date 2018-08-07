# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
TensorFlow 1.X Recipes for Artificial Intelligence Applications
Section 4
2 - A simple environment and basic policies
"""
#%% imports
import gym
import numpy as np
import pandas as pd
#%% Environment
env = gym.make('CartPole-v1')

# actions
print('Action space:', env.action_space)

# observation
observation = env.reset()
obs_dict = {
        0: 'Cart Position',
        1: 'Cart Velocity',
        2: 'Pole Angle',
        3: 'Pole Velocity At Tip'}

for i in range(4):
    print(obs_dict[i],':' ,observation[i])
    

#%% one step
action = 1 #push right
observation, reward, done, info = env.step(action)

print('State:')
for i in range(4):
    print('\t', obs_dict[i], ':', observation[i])

print('Reward:', reward)
print('Done:', done)
print('Info:', info)

#%% a random agent
n_episodes = 30
max_steps_per_episode = 200
for i_episode in range(n_episodes):
    observation = env.reset()
    for t in range(max_steps_per_episode):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode {} finished after {} timesteps".format(i_episode+1, t+1))
            break
env.close()
#%% Defining simple agents

def leftist_agent(observation):
    '''Always goes left'''
    return 0

def random_agent(observation):
    '''Samples from the possible actions, all equally likely'''
    action = env.action_space.sample()
    return action

def angle_agent(observation):
    '''Looks at the Pole Angle and goes in that direction'''
    pole_angle = observation[2]
    if pole_angle > 0:
        action = 1
    else:
        action = 0
    return action

def center_agent(observation):
    '''Looks at the Cart Position from the center and moves the other way'''
    position = observation[1]
    if position > 0:
        action = 0
    else:
        action = 1
    return action


#%% Comparing the 4 agents
agents = [leftist_agent, random_agent, angle_agent, center_agent]

n_episodes = 1000
max_steps_per_episode = 200
agent_steps = {}
for agent in agents:
    steps_per_episode = []
    for i_episode in range(n_episodes):
        observation = env.reset()
        for step in range(max_steps_per_episode):
            action = agent(observation)
            observation, reward, done, info = env.step(action)
            if done:
                steps_per_episode.append(step+1)
                break
    agent_steps[agent.__name__] = np.array(steps_per_episode)       

df = pd.DataFrame(agent_steps)

df.describe()