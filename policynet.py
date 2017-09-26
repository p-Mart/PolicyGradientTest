
import gym
import numpy as np
import random
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
from keras import optimizers

#Hyperparameters
h_size = 200
batch_size = 10
learning_rate = 1e-3
gamma = 0.99
decay_rate = 0.99
resume = False
render = False

#input dimensions
D = 80**2


def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(xrange(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


#Initialize weights with xavier
inputs = Input(shape=(6400,))
x = Dense(h_size, activation="relu")(inputs)
output = Dense(1, activation="sigmoid")(x)
xs, targets, rewards = [], [], []
model = Model(inputs=inputs, outputs=output)
rmsprop = optimizers.RMSprop(lr=learning_rate, decay=decay_rate)

model.compile(optimizer="Adam", loss="mean_squared_error")

env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None

running_reward = None
reward_sum = 0
episode_number = 0

while True:

    render = True if (episode_number+1 % (batch_size*20) == 0) else False

    if render : env.render()

    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    #Sample policy network
    aprob = model.predict(x.reshape((1, 80*80)))
    action = 2 if np.random.uniform() < aprob else 3

    xs.append(x)

    y = 1 if action == 2 else 0

    targets.append(y - aprob)

    observation, reward, done, info = env.step(action)
    reward_sum += reward

    rewards.append(reward)

    if done:
        episode_number += 1

        epx = np.vstack(xs)
        eptargets = np.vstack(targets)
        eprewards = np.vstack(rewards)


        xs, targets, rewards = [], [], []

        discounted_epreward = discount_rewards(eprewards)

        #Unit normal standardization
        discounted_epreward -= np.mean(discounted_epreward)
        discounted_epreward /= np.std(discounted_epreward)
        eptargets *= discounted_epreward

        if episode_number % batch_size == 0:
            model.fit(epx, eptargets, epochs=1, batch_size=batch_size)

        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print 'resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward)
        observation = env.reset()
        reward_sum = 0
        prev_x = None
