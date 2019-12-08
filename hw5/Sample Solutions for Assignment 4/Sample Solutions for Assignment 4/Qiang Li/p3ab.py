# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 14:58:56 2018

@author: qli14
"""
import numpy as np

def Gt_return(state, reward, gamma):
    Gt = 0
    for i in range(1, len(state)):
        Gt += reward[i] * np.power(gamma, i-1) 
    return Gt

np.random.seed(1)
state = np.random.randint(2, size=10000)
state[0] = 0 # start state
reward = np.zeros(state.shape)
# last state = 0, then reward = 1
# last state = 1, then reward = 2
reward[1:] = state[0:-1] + 1
print(state)

gamma = 0.75

#print(Gt_return(state[:5], reward[:5], gamma))
V0 = 0
V1 = 1
N0 = 0
N1 = 1
for i in range(len(state)):
    if state[i] == 0:
        N0 += 1
        Gt = Gt_return(state[i:], reward[i:], gamma)
        V0 = V0 + 1.0/N0 * ( Gt - V0 )
    else:
        N1 += 1
        Gt = Gt_return(state[i:], reward[i:], gamma)
        V1 = V1 + 1.0/N1 * ( Gt - V1 )
    if i%100 == 0:
        print("i = %d, V0 = %0.3f, V1 = %0.3f" % (i, V0, V1))
