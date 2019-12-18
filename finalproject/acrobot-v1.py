import random
import gym
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
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = 0.05
        self.batch_size = 32
        # Create model
        # Two seperate models, one for doing predictions 
        # and other for tracking target values
        self.model = self.create_model()
        self.target_model = self.create_model()

    # Neural Network model for function approximation
    def create_model(self):
        model   = Sequential()
        state_shape  = self.env.observation_space.shape
        model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
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
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)

    # Chooses action
    def action(self, state):
        # Update the epsilon to lower  exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
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

def main():
    env = gym.make('Acrobot-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    DQNAgent = DQN(env)
    done = False
    trials = 500
    trial_len = 1000

    for trial in range(trials):
        # reset environment
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        for step in range(trial_len):
            env.render()
            # Choose action
            action = DQNAgent.action(state)
            # Get next state and reward
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            DQNAgent.save(state, action, reward, next_state, done)

            # Train both Neural Networks
            DQNAgent.train()
            DQNAgent.target_train()
            
            # Update state
            state = next_state
            if done:
                print(f"Trial: {trial}, Score: {trial_len-step}, e: {DQNAgent.epsilon:.4} ")
                break



if __name__ == "__main__":
    main()





