# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 21:54:58 2018

@author: qli14
"""
import numpy as np
def q0(a, v0, v1):
    ## Input: the action taken, a = 1 or 2
    ##        v0: v*(0)
    ##        v1: v*(1)
    ## Output: q*(s=0, a)
    gamma = 3.0/4.0
    if a == 1:
        R = 1
        # to state 0
        P0 = 1.0/3.0
        P1 = 1 - P0
    else:
        R = 4
        P0 = 1.0/2.0
        P1 = 1- P0
    vstar = R + gamma * ( P0 * v0 + P1 * v1)
    return vstar

def q1(a, v0, v1):
    ## Input: the action taken, a = 1 or 2
    ##        v0: v*(0)
    ##        v1: v*(1)
    ## Output: q*(s=1, a)
    gamma = 3.0/4.0
    if a == 1:
        R = 3
        # to state 0
        P0 = 1.0/4.0
        P1 = 1 - P0
    else:
        R = 2
        P0 = 2.0/3.0
        P1 = 1- P0
    vstar = R + gamma * ( P0 * v0 + P1 * v1)
    return vstar

v0 = 0
v1 = 0
for i in range(0,50):
#    print("Step ", i)
    v0_temp = np.maximum(q0(1, v0, v1), q0(2, v0, v1))
    v1 = np.maximum(q1(1, v0, v1), q1(2, v0, v1))
    v0 = v0_temp
    print("step: %d, v0 = %0.3f, v1 = %0.3f" % (i, v0, v1) )

# q*(0, 1), q*(0, 2), q*(1, 1), q*(1, 2)
q01 = q0(1, v0, v1)
q02 = q0(2, v0, v1)
q11 = q1(1, v0, v1)
q12 = q1(2, v0, v1)

