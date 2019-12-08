# -*- coding: utf-8 -*-
"""
## TD for value function estimation

Created on Sat Dec  1 14:58:56 2018

@author: qli14
"""
import numpy as np

def Gt_return(state, reward, gamma):
    Gt = 0
    for i in range(1, len(state)):
        Gt += reward[i] * np.power(gamma, i-1) 
    return Gt

def nStepTD(state, reward, gamma, n, V0, V1):
    N0 = 0
    N1 = 1
    for i in range(len(state)-n): # -1 because I need to check V(S_{t+1})
        if state[i] == 0:
            N0 += 1
            # n-step return
            Gt_n = Gt_return(state[i:i+n+1], reward[i:i+n+1], gamma)
            if state[i+n] == 0:
                V_next = V0
            else:
                V_next = V1
            V0 = V0 + 1.0/N0 * ( Gt_n + np.power(gamma, n) * V_next - V0 )
        else:
            N1 += 1
            # n-step return
            Gt_n = Gt_return(state[i:i+n+1], reward[i:i+n+1], gamma)
            if state[i+n] == 0:
                V_next = V0
            else:
                V_next = V1
            V1 = V1 + 1.0/N1 * ( Gt_n + np.power(gamma, n) * V_next - V1 )
    if i%100 == 0:
        print("i = %d, V0 = %0.3f, V1 = %0.3f" % (i, V0, V1))
    return V0, V1

np.random.seed(1)
state = np.random.randint(2, size=10000)
state[0] = 0 # start state
reward = np.zeros(state.shape)
# last state = 0, then reward = 1
# last state = 1, then reward = 2
reward[1:] = state[0:-1] + 1
#print(state)

gamma = 0.75

#print(Gt_return(state[:5], reward[:5], gamma))
V0 = 0
V1 = 1
for n in range(1,6):
    (V0, V1) = nStepTD(state, reward, gamma, n, V0, V1)
    print("n = %d, V0 = %0.3f, V1 = %0.3f" %(n, V0, V1))

