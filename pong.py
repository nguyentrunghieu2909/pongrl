import ale_py
from ale_py import ALEInterface
ale = ALEInterface()

import gymnasium as gym


gym.register_envs(ale_py)

# env = gym.make('ALE/Breakout-v5')
# obs, info = env.reset()
# obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
# env.close()

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten
from keras.optimizers import Adam
from keras.layers import Conv2D

import sys


class PGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self._build_model()
        self.model.summary()

    def _build_model(self):
        model = Sequential()
        model.add(Reshape((1, 80, 80), input_shape=(self.state_size,)))
        model.add(Conv2D(32, kernel_size=(6, 6), strides=(3, 3), padding='same',
                          activation='relu', kernel_initializer='he_uniform'))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, state):
        state = state.reshape([1, state.shape[0]])
        aprob = self.model(state, training=False).numpy().flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards = rewards / np.std(rewards - np.mean(rewards))
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]))
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []

    def load(self, name):
        self.model.built = True
        self.model.load_weights(name)

    def save(self, name):
        self.model.save(name)


def preprocess(I):
    # Convert the 210x160x3 uint8 frame into a 6400 float vector
    I = I[35:195] # crop playing area
    I = I[::2, ::2, 0] # Take only alternate pixels and halves the resolution of the image & convert all color
    I[I == 144] = 0 # remove background color
    I[I == 109] = 0 # remove another shade of background
    I[I != 0] = 1 # convert paddles and ball into 1
    return I.astype(np.float32).ravel() # convert image into an 1d array of float


if __name__ == "__main__":
    env = gym.make("ALE/Pong-v5", render_mode="human")
    state, _ = env.reset()
    prev_x = None
    score = 0
    episode = 0

    state_size = 80 * 80
    action_size = env.action_space.n
    agent = PGAgent(state_size, action_size)
    # try catch load previous model
    agent.load('./save_model/pong_reinforce.h5')
    while True:
        env.render()

        cur_x = preprocess(state)
        x = cur_x - prev_x if prev_x is not None else np.zeros(state_size)
        prev_x = cur_x

        action, prob = agent.act(x)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
        agent.remember(x, action, prob, reward)

        if done:
            episode += 1
            # agent.train()
            print(f'Episode: {episode} - Score: {score:.2f}')
            score = 0
            state, _ = env.reset()
            prev_x = None
            if episode > 1 and episode % 10 == 0:
                agent.save('./save_model/pong_reinforce.h5')
            # if episode == 100:
            #     sys.exit(0)
