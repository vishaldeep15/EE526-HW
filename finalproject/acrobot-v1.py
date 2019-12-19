import random
import gym
import time
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DQN:
    # Initialize env and all parameters
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.epsilon = 0.5
        self.epsilon_min = 0.001
        self.epsilon_decay = 0.9992
        self.learning_rate = 1e-4
        self.batch_size = 64
        self.tau = 1e-3
        # Create model
        # Two seperate models, one for doing predictions 
        # and other for tracking target values
        self.model = self.create_model()
        self.target_model = self.create_model()

    # Neural Network model for function approximation
    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(5, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(10, activation="relu"))
        # model.add(Dense(5, activation="relu"))
        model.add(Dense(self.env.action_space.n))
        model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
        return model

    # add learned states,action, and rewards to the list
    def save(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def train(self):
        # if memory is smaller then do nothing
        if len(self.memory) < self.batch_size:
            return
        
        # Take a random samples from the memory
        samples = random.sample(self.memory, self.batch_size)

        for sample in samples:
            state, action, reward, next_state, done = sample
            target = self.target_model.predict(state)
            # If at the end of trials, there are no future rewards
            if done:
                target[0][action] = reward
            else:
                Q_future = np.max(self.target_model.predict(next_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            # Fit the model
            self.model.fit(state, target, epochs=1, verbose=0)

    # Updates the weights in target NN
    def target_train(self):
        # Get learned weights
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)
        # Update the epsilon to lower  exploration rate
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

    # Chooses action
    def action(self, state):
        # # Update the epsilon to lower  exploration rate
        # self.epsilon *= self.epsilon_decay
        # self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.rand() < self.epsilon:
            #Choose random action
            return self.env.action_space.sample()
        # Else choose max value action
        return np.argmax(self.model.predict(state)[0])

    # Load weights
    def load_weights(self, name):
        self.model.load_weights(name)
    # Save weights
    def save_weights(self, name):
        self.model.save_weights(name)

def main(render=False):
    env = gym.make('Acrobot-v1')
    # state_size = env.observation_space.shape[0]
    # # action_size = env.action_space.n
    DQNAgent = DQN(env)
    state_size = DQNAgent.state_size
    done = False
    episodes = 500
    trial_len = 500

    # Print Neural Network model summary
    DQNAgent.model.summary()

    # Capture training start time
    startTime = np.round(time.time(), decimals=4)
    for episode in range(episodes):
        rewards = []
        # reset environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for step in range(trial_len):
            if render:
                env.render()
            # Choose action
            action = DQNAgent.action(state)
            # Get next state and reward
            next_state, reward, done, _ = env.step(action)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            DQNAgent.save(state, action, reward, next_state, done)

            # Train prediction Neural Networks
            DQNAgent.train()
            # Update target NN periodically
            if step%10 == 0:
                # print("Updating target Network")
                DQNAgent.target_train()
            # Append Rewards
            rewards.append(reward)            
            # Update state
            state = next_state
            if done:
                print("episode: {}, score: {}, epsilon: {:.6}, Rewards: {}"
                      .format(episode, trial_len-step, DQNAgent.epsilon, np.sum(rewards)))
                # print(f"Episode:{episode},Score:{trial_len-step}/{trial_len},Epsilon:{DQNAgent.epsilon}")
                break
    # Calculate time taken to train      
    stopTime = np.round(time.time(), decimals=4)
    totalTime = (np.round(((stopTime - startTime)/60), decimals=4))
    print('Training time: {} minutes'.format(totalTime))

if __name__ == "__main__":
    main()