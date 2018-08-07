# -*- coding: utf-8 -*-
"""
@author: Alvaro Fuentes
TensorFlow 1.X Recipes for Artificial Intelligence Applications
Section 4
3 - Training a neural network policy
Based mainly on https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb
"""
#%% imports
import gym
import tensorflow as tf
import numpy as np

#%% environment
env = gym.make("CartPole-v1")

#%% Neural Network Policy

n_inputs = 4
observation = tf.placeholder(tf.float32, shape=[None, n_inputs])
def neural_network(observation):
    '''Takes a set of observations and returns the logits for the two possible actions'''
    n_hidden = 8
    n_outputs = 1
    initializer = tf.contrib.layers.variance_scaling_initializer()
    hidden = tf.layers.dense(observation, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
    output = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
    return output

#%% Choosing an action: based on the probabilites
## equivalent to: np.random.choice
logits = neural_network(observation)    
proba_left = tf.nn.sigmoid(logits)
probabilities = tf.concat(axis=1, values=[proba_left, 1 - proba_left])
log_probs = tf.log(probabilities)
action = tf.multinomial(log_probs, num_samples=1)

## Target probability: we are acting as though the chosen action
## is the best possible action, the target probability must be 1.0
## if the chosen action is action 0 (left) and 0.0 if it is action 1 (right):
y = 1. - tf.to_float(action)

#%% loss & optimizer
learning_rate = 0.01
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)

#%% getting the gradients
# returns a pair (gradient, variable)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
# producing a list of gradients for later processing
gradients = [grad for grad, variable in grads_and_vars]

#%% Placeholders for processing gradients before applying them
# We need a list of placeholders for the gradients...
gradient_placeholders = []

# ...and a list of gradients and corresponding variables to apply them
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
    
#%% The trainning operation consists on applying gradients
training_op = optimizer.apply_gradients(grads_and_vars_feed)

#%% Discounting and normalizing rewards

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
            for discounted_rewards in all_discounted_rewards]

#%% Training the network
n_iterations = 250
episodes_per_update = 10 # train the policy every 10 episodes
save_iterations = 20    # save the model every 20 training iterations
discount_rate = 0.95

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration+1), end="")
        all_rewards = []    # all sequences of raw rewards for each episode
        all_gradients = []  # gradients saved at each step of each episode
        for episode in range(episodes_per_update):
            episode_rewards = []
            episode_gradients = []
            obs = env.reset()
            done = False
            while not done:
                action_val, gradients_val = sess.run([action, gradients],
                                                     feed_dict={observation: obs.reshape(1, n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                episode_rewards.append(reward)
                episode_gradients.append(gradients_val)
            ## When episode ends save rewards and gradients
            all_rewards.append(episode_rewards)
            all_gradients.append(episode_gradients)

        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
        feed_dict = {}
        # for each of the trainable variables...
        for var_index, gradient_placeholder in enumerate(gradient_placeholders):
            # for each of the 10 episodes...
            for episode_index, rewards in enumerate(all_rewards):
                # get the reward of every step, and use it to modify the gradient
                processed_gradients = [reward * all_gradients[episode_index][step][var_index]
                                              for step, reward in enumerate(rewards)]
            # calculate the mean of the processed gradients and use that value in the placeholder
            feed_dict[gradient_placeholder] = np.mean(processed_gradients, axis=0)
        sess.run(training_op, feed_dict=feed_dict)
        if iteration % save_iterations == 0:
            saver.save(sess, "./SavedModels/policy_network.ckpt")
    print("\nDone!")
