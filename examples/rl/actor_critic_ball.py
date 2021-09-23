"""
Title: Actor Critic Method
Author: [Apoorv Nandan](https://twitter.com/NandanApoorv)
Date created: 2020/05/13
Last modified: 2020/05/13
Description: Implement Actor Critic Method in CartPole environment.
"""
"""
## Introduction

This script shows an implementation of Actor Critic method on CartPole-V0 environment.

### Actor Critic Method

As an agent takes actions and moves through an environment, it learns to map
the observed state of the environment to two possible outputs:

1. Recommended action: A probability value for each action in the action space.
   The part of the agent responsible for this output is called the **actor**.
2. Estimated rewards in the future: Sum of all rewards it expects to receive in the
   future. The part of the agent responsible for this output is the **critic**.

Agent and Critic learn to perform their tasks, such that the recommended actions
from the actor maximize the rewards.

### CartPole-V0

A pole is attached to a cart placed on a frictionless track. The agent has to apply
force to move the cart. It is rewarded for every time step the pole
remains upright. The agent, therefore, must learn to keep the pole from falling over.

### References

- [CartPole](http://www.derongliu.org/adp/adp-cdrom/Barto1983.pdf)
- [Actor Critic Method](https://hal.inria.fr/hal-00840470/document)
"""
"""
## Setup
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from examples.rl.env.ball import BallEnv

# Configuration parameters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
# max_steps_per_episode = 10000
max_steps_per_episode = 100
# env = gym.make("CartPole-v0")  # Create the environment
# env = gym.make("ball")
env = BallEnv()
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

"""
## Implement Actor Critic network

This network learns two functions:

1. Actor: This takes as input the state of our environment and returns a
probability value for each action in its action space.
2. Critic: This takes as input the state of our environment and returns
an estimate of total rewards in the future.

In our implementation, they share the initial layer.
"""

# num_inputs = 4
# num_actions = 2
num_actions = 4
# num_hidden = 128
dim_input = [env.space_x_max, env.space_y_max, env.input_num_channel]
# max_sdev = 5.0

# inputs = layers.Input(shape=(num_inputs,))
inputs = layers.Input(shape=dim_input)
# common = layers.Dense(num_hidden, activation="relu")(inputs)
common_conv2d_1 = layers.Conv2D(filters=16, kernel_size=8, strides=4, padding="valid", activation="relu")(inputs)
common_conv2d_2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, padding="valid", activation="relu")(common_conv2d_1)
common_flatten = layers.Flatten()(common_conv2d_2)
common_fc_1 = layers.Dense(units=256, activation="relu")(common_flatten)
common_fc_2 = layers.Dense(units=256, activation="relu")(common_fc_1)
# action = layers.Dense(num_actions, activation="softmax")(common)

# actor_mean_tanh = layers.Dense(units=2, activation="tanh")(common_fc_2)
# actor_mean = layers.Rescaling(scale=env.max_force, offset=0.0)(actor_mean_tanh)
# actor_sdev_sig = layers.Dense(units=2, activation="sigmoid")(common_fc_2)
# actor_sdev = layers.Rescaling(scale=max_sdev, offset=0.0)(actor_sdev_sig)
# action = extract(action_mean, action_sdev)
action = layers.Dense(num_actions, activation="softmax")(common_fc_2)
# actor = layers.Concatenate(axis=-1)([actor_mean, actor_sdev])
critic = layers.Dense(1)(common_fc_2)

model = keras.Model(inputs=inputs, outputs=[action, critic])
# model = keras.Model(inputs=inputs, outputs=[actor, critic])
model_jason = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_jason)
"""
## Train
"""

optimizer = keras.optimizers.Adam(learning_rate=0.01)
huber_loss = keras.losses.Huber()
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:  # Run until solved
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            # print(timestep)
            # if episode_count % 100 == 0:
            if episode_count % 1 == 0:
                env.render() # Adding this line would show the attempts
            # of the agent in a pop up window.

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            # Predict action probabilities and estimated future rewards
            # from environment state
            action_probs, critic_value = model(state)
            # action_meanNsdev, critic_value = model(state)
            # action_mean = action_meanNsdev[0, 0:2]
            # action_sdev = action_meanNsdev[0, 2:4]
            critic_value_history.append(critic_value[0, 0])

            # Sample action from action probability distribution
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))
            # action_x_mean = action_mean[0]
            # action_y_mean = action_mean[1]
            # action_x_sdev = action_sdev[0]
            # action_y_sdev = action_sdev[1]
            # action_x = np.random.normal(loc=action_x_mean, scale=action_x_sdev)
            # action_y = np.random.normal(loc=action_y_mean, scale=action_y_sdev)
            # action = [action_x, action_y]
            action_probs_history.append(tf.math.log(action_probs[0, action]))
            # action_x_prob = (1/(action_x_sdev*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((action_x-action_x_mean)/action_x_sdev)**2)
            # action_y_prob = (1/(action_y_sdev*np.sqrt(2 * np.pi))) * np.exp((-1/2) * ((action_y-action_y_mean)/action_y_sdev) ** 2)
            # action_probs_history.append(tf.math.log([action_x_prob, action_y_prob]))

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward

            if done:
                break

        # Update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0, discounted_sum) # returns: [G1, G2, ...]

        # Normalize
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()

        # Calculating loss values to update our network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff)  # actor loss

            # The critic must be updated so that it predicts a better estimate of
            # the future rewards.
            critic_losses.append(
                huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0))
            )

        # Backpropagation
        loss_value = sum(actor_losses) + sum(critic_losses)
        # loss_value = sum(sum(actor_losses)) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Clear the loss and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_count))

    # Save the weights
    if episode_count % 100 == 0:
        model.save_weights("model" + str(episode_count) + ".h5")
        print("Saved model to disk")

    # if running_reward > 195:  # Condition to consider the task solved
    #     print("Solved at episode {}!".format(episode_count))
    #     break

"""
## Visualizations
In early stages of training:
![Imgur](https://i.imgur.com/5gCs5kH.gif)

In later stages of training:
![Imgur](https://i.imgur.com/5ziiZUD.gif)
"""
