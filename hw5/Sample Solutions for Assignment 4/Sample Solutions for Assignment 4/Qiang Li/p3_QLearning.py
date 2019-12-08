# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 19:24:12 2018
##
Use Q-Learning to estimate optimal action value function
@author: qli14
"""
import numpy as np
## Initialize action-value function
## q(s, a)
q = np.array([
        [0., 0],
        [0, 0]
        ])
#q = np.random.rand(2, 2)
## Reward(s, a)
reward = np.array([
        [1., 4.],
        [3., 2.]
        ])
## epsilon-greedy
epsilon = 0.1

## step size
alpha = 0.4

## discount factor
gamma = 0.75

#np.random.seed(1)

## Initialize state, 0 or 1
state = np.random.randint(2)
## Choose action from state with epsilon-greedy
if np.random.rand() < epsilon:
    # Explore, action = 1 or 2
    action = np.random.randint(2)
else:
    # Choose action with larger q value
    # index is 0 and 1 while action is 1 and 2
    action = np.argmax(q[state, :])

## Repeat for each step of episode    
for i in range(500000): 
    R = reward[state, action]
    state_next = np.random.randint(2) ## How to determine next state?
    ## Choose action_next with epsilon-greedy
    if np.random.rand() < epsilon:
        action_next = np.random.randint(2)
    else:
        action_next = np.argmax(q[state_next, :])
#    print(q)
    ## update q value    
    q[state, action] += alpha * ( R + 
                 gamma * np.max(q[state_next, :]) - q[state, action] )
#    print("alpha = %0.3f" %alpha)
    if i%1000 == 0:
        print("i = %d \n (s,a) = (%d,%d), (s',a') = (%d,%d)"
              % (i, state, action, state_next, action_next))
        print("q = \n (%0.3f, %0.3f) \n (%0.3f, %0.3f)" 
              %(q[0,0],q[0, 1],q[1,0], q[1,1]))
   
    state = state_next
    action = action_next
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        