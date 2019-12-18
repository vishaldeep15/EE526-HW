# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 13:20:21 2018

@author: mailr
"""
'''

The code was written by referencing the cartpole code available in the following link on the web:
https://github.com/gsurma/cartpole/blob/master/cartpole.py

'''

# importing required packages
from gym import wrappers
from time import time
import math
import random # importing random to be used for implementing greedy-E policy
import gym # importing gym to use the acrobot-v1 environment
import numpy as np #numpy import
from collections import deque
from keras.models import Sequential # using keras to build the deep learning network
from keras.layers import Dense
from keras.optimizers import Adam

import matplotlib.pyplot as plt # using plt to plot the total_reward as a function of number of episodes

#initialising empty list to store data
xdata = [] # It will store the episode number
ydata = [] # It will store the total reward accumulated from random initial state to a terminal state in an episode
 
# Initialising the environment as Acrobot-v1 taken from open AI gym
ENV_NAME = "Acrobot-v1"

# The discount factor
GAMMA = 1 

# The learning rate of our neural network
LEARNING_RATE = 0.001 

# The memory to store batches of state-action pairs
MEMORY_SIZE = 1000000 

# The batch size
BATCH_SIZE = 32

# To implement the E greedy policy, We are chosing a decaying exploration E factor with initial value as 1 and decaying to EXPLORATION_MIN
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.02
EXPLORATION_DECAY = 0.995

# Initialising list  to store the total reward history
total_reward_history=[]

Neurons_Layer_1 = 8
Neurons_Layer_2 = 8

# Deep Q learning Network
class DeepQLearningNetwork:

#Architecture of the Deep Neural Network used as the  function approximator
    def __init__(self, observation_space, action_space):
        #intialising with the maximum exploration rate
        self.exploration_rate = EXPLORATION_MAX 

        # Action space is needed to determine how many final outputs are needed from the deep neural network
        self.action_space = action_space 
        
        # Memory allocation
        self.memory = deque(maxlen=MEMORY_SIZE) 

        self.model = Sequential()
        
        # First layer of Neural network with input as the number of values required to determine a state given by observation_space
        self.model.add(Dense(Neurons_Layer_1, input_shape=(observation_space,), activation="relu"))
        
        # Second Layer of the Neural Network with number of neurons as  Neurons_Layer_2
        self.model.add(Dense(Neurons_Layer_2, activation="relu"))
        
        # Final Output layer with the number of outputs equal to the number of possible action spaces i.e 3 in case of acrobot-v1 +1,0,-1
        self.model.add(Dense(self.action_space, activation="linear"))
        
        #Using adam optimizer with mean square error
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        
        
 # Remembering the current state, action, reward, next state and if the state is terminal into the memory for Batch learning of the Q network
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# Deciding on what action to take on the current state using a exploration-greedy policy
    def act(self, state):
        #Randomly choosing an action
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        #Using the learned model to predict the best action to chose by using the max (Q(S,A))
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

# Batch Learning, the deep learning network replay a random sample of Batch_Size choosen from the remembered memory and learns
    def experience_replay(self):
        # The memory has to reach atleast the batch size for learning
        if len(self.memory) < BATCH_SIZE:
            return
        #Sampling out random batch of BATCH_SIZE
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            #print(state_next)
            #Modified the reward here as -cos(theta_1)-cos(theta_1+theta_2)
            # To calculate theta_1 we have to take cos inverse of state[0] as state[0] stores the value of cos(theta_1)
            #To calculate theta_2 we have to take cos inverse of state[2] as state[2] stores the value of cos(theta_2)
            # Inspiration here is taken from the check of terminal state i.e. -np.cos(s[0]) - np.cos(s[1] + s[0]) > 1 
            # And so more reward should be given as -cos(theta_1)-cos(theta_1+theta_2) increases
            # I am trying to make the reward here proportional to -cos(theta_1)-cos(theta_1+theta_2)
            Q1 = math.acos((state_next[0][0]))
            Q2 = math.acos((state_next[0][2]))
            reward  = -math.cos(Q1)-math.cos(Q1+Q2)
            q_update = reward
            if not terminal:
                #This the core equation of deep Q learning which uses immediate reward plus discount factor * max(Q(s,a)) predicted by the model
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            # Deep NN learns the calculated Q values for the state
            self.model.fit(state, q_values, verbose=0)
        
        # Exploration Decay
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


# Simulating Episodes
def acrobot():
    env = gym.make(ENV_NAME).env # initialising env as acrobot
    env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
  
    observation_space = env.observation_space.shape[0] # No of state values
    action_space = env.action_space.n # No of Action possible
    dqn_solver = DeepQLearningNetwork(observation_space, action_space) # Initialising the DQN
    episode = 0 # variable to store episode number
    total_training_steps = 0 # Variable to store total training steps till desired accuracy is achieved
   
    
    while episode<=100:
        episode += 1          # Incrementing episode
        xdata.append(episode) # Storing the episode number to plot
        total_reward=0        # Variable to store the total accumulated reward of the current episode
        state = env.reset()   #Picks up a random current state from the env
       
        state = np.reshape(state, [1, observation_space]) # Picks up the state observation space
        step = 0              # stores the count of number of steps in current episode
        while True:
            step += 1
            env.render() # renders the moving pole on the dislay to visualize
            action = dqn_solver.act(state) # Chooses the best action possible on the current state using the deep neural network
            
            state_next, reward, terminal, info = env.step(action) # takes the action generated above and calculates next state, reward, if the state is terminal etc
            
            # As per the implementation of Acrobot-v1 in git the reward is -1 if state is not terminal and 0 if terminal
            # So the reward of -1 puts incentive on reaching the terminal state fast
           
            reward = reward if not terminal else -reward
            
            # Accumulating the total reward
            total_reward+=reward
            
            
            state_next = np.reshape(state_next, [1, observation_space])
            
            dqn_solver.remember(state, action, reward, state_next, terminal) # remembering the values in memory
            state = state_next # updating current state by the next state
          
            # End of Episode if the terminal state is reached
            if terminal:
                
                print ('episode: '+ str(episode) + ", exploration: " + str(dqn_solver.exploration_rate) + ", steps: " + str(step) + ' Total Reward is: ' + str(total_reward ))
                ydata.append(total_reward) # storing total_reward accumulated till the end of episode to plot
                total_reward_history.append(total_reward) # Storing the total_ accumulated reward in the history
                
                total_training_steps+=step # Accumulating steps 
                break
            else:
               
                dqn_solver.experience_replay() # DQN learning if not terminal
        
    '''    if ( np.mean(total_reward_history) >= -115): # Checking convergenes if the average reward reaches -120
          
            print('Total steps required to train the network to give reasonable accuracy is '+ str(total_training_steps))
            return
        '''

if __name__ == "__main__":
    acrobot()
    axes = plt.gca()
    axes.set_xlim([1, 100])
    plt.plot(xdata,ydata)
    plt.xlabel('Simulated Episode number')
    plt.ylabel('Total Reward')
    plt.show()
    print('Total steps required to train the network to give reasonable accuracy is '+ str(acrobot()))
    print('Average total reward obtained per episode was '+(np.mean(total_reward_history)))
    print('Maximum best total reward obtained was '+(np.max(total_reward_history)))