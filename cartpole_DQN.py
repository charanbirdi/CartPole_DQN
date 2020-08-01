import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop
import numpy as np
import random
from keras.models import Model, load_model


def AImodel(input_shape, action_space):
    model = Sequential()
    model.add(Dense(128, input_dim=input_shape, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(action_space, activation='linear', kernel_initializer='he_uniform'))
    model.compile(loss = "mse", optimizer="RMSprop", metrics=["accuracy"])
    model.summary()
    return model


class Dqnagent():
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.EPISODE = 1000
        self.memory = deque(maxlen=2000)
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.001
        self.train_start = 1000
        self.batch_size = 64
        self.gamma = 0.95

        self.model = AImodel(input_shape=self.state_size, action_space=self.action_size)

    def act(self, state):
        if np.random.random() <= self.epsilon:
            #print("Exploreation")
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.train_start:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay

    def save(self, name):
        self.model.save(name)

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        state2, target2 = [], []
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            target[0][action] = reward if done else reward + np.amax(self.model.predict(next_state)[0])
            state2.append(state[0])
            target2.append(target[0])            
        
        self.model.fit(np.array(state2), np.array(target2), batch_size = len(minibatch), verbose=0)
        

    def run(self):
        for e in range(self.EPISODE):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i=0
            while not done:
                #self.env.render()
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done or i == self.env._max_episode_steps-1:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i = i+1
                if done:
                    print("Episode: {}/{}, score: {}, e: {}".format(e, self.EPISODE, i, self.epsilon))
                    if i == 500:
                        print("Saving Trained Model as cartpole_dqn.h5")
                        self.save("cartpole_dqn.h5")
                        return
                self.replay()


    def load(self, name):
        self.model = load_model(name)

    

    def test(self):
        self.load("cartpole_dqn.h5")
        for e in range(self.EPISODE):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i = i + 1
                if done:
                    print("episode: {}/{}, score: {}".format(e, self.EPISODE, i))
                    break


if __name__=="__main__":
    agent  = Dqnagent()
    #agent.run()
    agent.test()