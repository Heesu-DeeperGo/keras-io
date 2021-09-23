"""
Title: Deep Deterministic Policy Gradient (DDPG)
Author: [amifunny](https://github.com/amifunny)
Date created: 2020/06/04
Last modified: 2020/09/21
Description: Implementing DDPG algorithm on the Inverted Pendulum Problem.
"""
"""
## Introduction

**Deep Deterministic Policy Gradient (DDPG)** is a model-free off-policy algorithm for
learning continous actions.

It combines ideas from DPG (Deterministic Policy Gradient) and DQN (Deep Q-Network).
It uses Experience Replay and slow-learning target networks from DQN, and it is based on
DPG,
which can operate over continuous action spaces.

This tutorial closely follow this paper -
[Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf)

## Problem

We are trying to solve the classic **Inverted Pendulum** control problem.
In this setting, we can take only two actions: swing left or swing right.

What make this problem challenging for Q-Learning Algorithms is that actions
are **continuous** instead of being **discrete**. That is, instead of using two
discrete actions like `-1` or `+1`, we have to select from infinite actions
ranging from `-2` to `+2`.

## Quick theory

Just like the Actor-Critic method, we have two networks:

1. Actor - It proposes an action given a state.
2. Critic - It predicts if the action is good (positive value) or bad (negative value)
given a state and an action.

DDPG uses two more techniques not present in the original DQN:

**First, it uses two Target networks.**

**Why?** Because it add stability to training. In short, we are learning from estimated
targets and Target networks are updated slowly, hence keeping our estimated targets
stable.

Conceptually, this is like saying, "I have an idea of how to play this well,
I'm going to try it out for a bit until I find something better",
as opposed to saying "I'm going to re-learn how to play this entire game after every
move".
See this [StackOverflow answer](https://stackoverflow.com/a/54238556/13475679).

**Second, it uses Experience Replay.**

We store list of tuples `(state, action, reward, next_state)`, and instead of
learning only from recent experience, we learn from sampling all of our experience
accumulated so far.

Now, let's see how is it implemented.
"""
import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from examples.rl.env.ball import BallEnv
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
"""
We use [OpenAIGym](http://gym.openai.com/docs) to create the environment.
We will use the `upper_bound` parameter to scale our actions later.
"""

# problem = "Pendulum-v0"
# env = gym.make(problem)
env = BallEnv()

# num_states = env.observation_space.shape[0]
dim_states = env.observation_space.shape
print("Size of State Space ->  {}".format(dim_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

"""
To implement better exploration by the Actor network, we use noisy perturbations,
specifically
an **Ornstein-Uhlenbeck process** for generating noise, as described in the paper.
It samples noise from a correlated normal distribution.
"""


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


"""
The `Buffer` class implements Experience Replay.

---
![Algorithm](https://i.imgur.com/mS6iGyJ.jpg)
---


**Critic loss** - Mean Squared Error of `y - Q(s, a)`
where `y` is the expected return as seen by the Target network,
and `Q(s, a)` is action value predicted by the Critic network. `y` is a moving target
that the critic model tries to achieve; we make this target
stable by updating the Target model slowly.

**Actor loss** - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to maximize this quantity.

Hence we update the Actor network so that it produces actions that get
the maximum predicted value as seen by the Critic, for a given state.
"""


class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        # self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.state_buffer = np.zeros((self.buffer_capacity, dim_states[0], dim_states[1], dim_states[2]))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        # self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.next_state_buffer = np.zeros((self.buffer_capacity, dim_states[0], dim_states[1], dim_states[2]))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        # self.action_buffer[index] = obs_tuple[1]
        self.action_buffer[index] = obs_tuple[1][0]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        # print(critic_grad)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))


"""
Here we define the Actor and Critic networks. These are basic Dense models
with `ReLU` activation.

Note: We need the initialization for last layer of the Actor to be between
`-0.003` and `0.003` as this prevents us from getting `1` or `-1` output values in
the initial stages, which would squash our gradients to zero,
as we use the `tanh` activation.
"""

kernel_size_block1 = 3
num_filters_block1 = 32

kernel_size_block2 = 3
num_filters_blick2 = 64

num_fc_units = 64

# dim_single_state = dim_states[1:]

def get_actor():
    # Initialize weights between -3e-3 and 3-e3
    last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=dim_states)
    # inputs = layers.Input(shape=dim_single_state)

    conv2d_block1_1 = layers.Conv2D(filters=num_filters_block1, kernel_size=kernel_size_block1, strides=1, padding="same")(inputs)
    conv2d_block1_2 = layers.Conv2D(filters=num_filters_block1, kernel_size=kernel_size_block1, strides=1, padding="same", activation="relu")(conv2d_block1_1)
    maxpool_block1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2d_block1_2)

    conv2d_block2_1 = layers.Conv2D(filters=num_filters_blick2, kernel_size=kernel_size_block2, strides=1, padding="same")(maxpool_block1)
    conv2d_block2_2 = layers.Conv2D(filters=num_filters_blick2, kernel_size=kernel_size_block2, strides=1, padding="same", activation="relu")(conv2d_block2_1)
    maxpool_block2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2d_block2_2)

    flatten = layers.Flatten()(maxpool_block2)

    fc1 = layers.Dense(num_fc_units, activation="relu")(flatten)
    fc2 = layers.Dense(num_fc_units, activation="relu")(fc1)

    outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(fc2)

    # out = layers.BatchNormalization()(out)
    # out = layers.Activation('relu')(out)
    # out = layers.Dense(32, activation="relu")(out)
    # out = layers.BatchNormalization()(out)
    # out = layers.Activation('relu')(out)
    # outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)`
    # outputs = layers.Dense(2, activation="tanh", kernel_initializer=last_init)(out)
    # outputs = layers.Dense(5, activation="softmax")(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * upper_bound
    model = tf.keras.Model(inputs, outputs)
    return model


def get_critic():
    # # State as input
    # state_input = layers.Input(shape=dim_states)
    # state_conv2d_1 = layers.Conv2D(filters=16, kernel_size=8, strides=4, padding="valid", activation="relu")(state_input)
    # # state_conv2d_1 = layers.BatchNormalization()(state_conv2d_1)
    # # state_conv2d_1 = layers.Activation('relu')(state_conv2d_1)
    # state_conv2d_2 = layers.Conv2D(filters=32, kernel_size=4, strides=2, padding="valid", activation="relu")(state_conv2d_1)
    # state_flatten = layers.Flatten()(state_conv2d_1)
    # # state_out = layers.Dense(16, activation="relu")(state_input)
    # # state_out = layers.Dense(32, activation="relu")(state_out)
    # state_out = layers.Dense(64, activation="relu")(state_flatten)
    # # state_out = layers.BatchNormalization()(state_out)
    # # state_out = layers.Activation('relu')(state_out)
    # state_out = layers.Dense(32, activation="relu")(state_out)
    # # state_out = layers.BatchNormalization()(state_out)
    # # state_out = layers.Activation('relu')(state_out)

    state_input = layers.Input(shape=dim_states)

    conv2d_block1_1 = layers.Conv2D(filters=num_filters_block1, kernel_size=kernel_size_block1, strides=1, padding="same")(state_input)
    conv2d_block1_2 = layers.Conv2D(filters=num_filters_block1, kernel_size=kernel_size_block1, strides=1, padding="same", activation="relu")(
        conv2d_block1_1)
    maxpool_block1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2d_block1_2)

    conv2d_block2_1 = layers.Conv2D(filters=num_filters_blick2, kernel_size=kernel_size_block2, strides=1, padding="same")(maxpool_block1)
    conv2d_block2_2 = layers.Conv2D(filters=num_filters_blick2, kernel_size=kernel_size_block2, strides=1, padding="same", activation="relu")(
        conv2d_block2_1)
    maxpool_block2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv2d_block2_2)

    flatten = layers.Flatten()(maxpool_block2)

    fc1 = layers.Dense(num_fc_units, activation="relu")(flatten)
    state_out = layers.Dense(num_fc_units, activation="relu")(fc1)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out = layers.Dense(num_fc_units, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    # out = layers.Dense(256, activation="relu")(concat)
    # out = layers.Dense(256, activation="relu")(out)
    out = layers.Dense(num_fc_units, activation="relu")(concat)
    out = layers.Dense(num_fc_units, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


"""
`policy()` returns an action sampled from our Actor network plus some noise for
exploration.
"""


def policy(state, noise_object):
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


"""
## Training hyperparameters
"""

# std_dev = 0.2
std_dev = 0.5
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

actor_model_jason = actor_model.to_json()
with open("actor_model.json", "w") as json_file:
    json_file.write(actor_model_jason)

critic_model_jason = critic_model.to_json()
with open("critic_model.json", "w") as json_file:
    json_file.write(critic_model_jason)

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Learning rate for actor-critic models
critic_lr = 0.001
actor_lr = 0.001
# actor_lr = 0.0001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)#, clipvalue=0.00001)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)#, clipvalue=0.00001)

total_episodes = 500
# Discount factor for future rewards
gamma = 0.99
# Used to update target networks
tau = 0.005

# buffer = Buffer(50000, 64)
buffer = Buffer(1000, 64)

"""
Now we implement our main training loop, and iterate over episodes.
We sample actions using `policy()` and train with `learn()` at each time step,
along with updating the Target networks at a rate `tau`.
"""

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []
episode_count = 0
max_steps_per_episode = 100

# Takes about 4 min to train
# for ep in range(total_episodes):
while True:

    prev_state = env.reset()
    episodic_reward = 0

    # while True:
    for timestep in range(1, max_steps_per_episode):
        # print(timestep)
        # Uncomment this to see the Actor in action
        # But not in a python notebook.
        # if episode_count % 100 == 0:
        if episode_count % 100 == 0:
            env.render()

        # prev_state_info = get_state_info(prev_state)
        # tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        if episode_count % 100 == 0:
            print(action)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(target_actor.variables, actor_model.variables, tau)
        update_target(target_critic.variables, critic_model.variables, tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    # print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    print("Episode * {} * Avg Reward is ==> {}".format(episode_count, avg_reward))
    avg_reward_list.append(avg_reward)

    episode_count += 1
    if episode_count % 100 == 0:
        actor_model.save_weights("ball_actor" + str(episode_count) + ".h5")
        critic_model.save_weights("ball_critic" + str(episode_count) + ".h5")
        target_actor.save_weights("ball_target_actor" + str(episode_count) + ".h5")
        target_critic.save_weights("ball_target_critic" + str(episode_count) + ".h5")
        print("Saved the weights")

# Plotting graph
# Episodes versus Avg. Rewards
# plt.plot(avg_reward_list)
# plt.xlabel("Episode")
# plt.ylabel("Avg. Epsiodic Reward")
# plt.show()

"""
If training proceeds correctly, the average episodic reward will increase with time.

Feel free to try different learning rates, `tau` values, and architectures for the
Actor and Critic networks.

The Inverted Pendulum problem has low complexity, but DDPG work great on many other
problems.

Another great environment to try this on is `LunarLandingContinuous-v2`, but it will take
more episodes to obtain good results.
"""

# Save the weights
# actor_model.save_weights("pendulum_actor.h5")
# critic_model.save_weights("pendulum_critic.h5")
#
# target_actor.save_weights("pendulum_target_actor.h5")
# target_critic.save_weights("pendulum_target_critic.h5")

"""
Before Training:

![before_img](https://i.imgur.com/ox6b9rC.gif)
"""

"""
After 100 episodes:

![after_img](https://i.imgur.com/eEH8Cz6.gif)
"""
