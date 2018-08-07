# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
Section 4
4 - Using neural network policy
"""
#%% imports
import gym
import tensorflow as tf
    
#%% environment
env = gym.make("CartPole-v1")
## Changing the limitation on steps
max_steps = 1000
env._max_episode_steps = max_steps
#%% Neural Net Policy
n_inputs = 4
observation = tf.placeholder(tf.float32, shape=[None, n_inputs])
def neural_network(observation):
    '''Takes an observation and returns the logits for the two possible actions'''
    n_hidden = 8
    n_outputs = 1
    initializer = tf.contrib.layers.variance_scaling_initializer()
    hidden = tf.layers.dense(observation, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    output = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    return output

def intelligent_agent(observation):
    logits = neural_network(observation)    
    proba_left = tf.nn.sigmoid(logits)
    probabilities = tf.concat(axis=1, values=[proba_left, 1 - proba_left])
    log_probs = tf.log(probabilities)
    return tf.multinomial(log_probs, num_samples=1)

action = intelligent_agent(observation)

saver = tf.train.Saver()

#%% using the agent
with tf.Session() as sess:
    saver.restore(sess, "./SavedModels/policy_network.ckpt")
    obs = env.reset()
    for step in range(max_steps):
        env.render()
        agent_action = sess.run(action, feed_dict={observation:obs.reshape(1,4)})
        obs, reward, done, info = env.step(agent_action[0][0])
        if done:
            break
    print("Total steps: ", step+1)

env.close() 