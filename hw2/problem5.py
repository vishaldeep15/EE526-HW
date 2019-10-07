# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:03:55 2019

@author: vishal
"""

from DNN import *
import numpy as np
import matplotlib.pyplot as plt

D = 57 # Input dimension
Odim = 1 # number of outputs
layers=[ (100, ReLU), (40, ReLU), (Odim, Linear) ]
    
# initialize Neural Network with D inputs and layers
nn = NeuralNetwork(D, layers)
# set random weights with maximum size of 0.1
nn.setRandomWeights(0.1)

# select crossentropy as objective function
CE = ObjectiveFunction('crossEntropyLogit')

eta = 1e-1

#print(X.shape)
##y.reshape(1, 60000)
#print(y.shape)
#print(y_test.shape)
#print(y)

for i in range(10000):
    logp = nn.doForward(X)
    J    = CE.doForward(logp, y)
    dz   = CE.doBackward(y)
    dx   = nn.doBackward(dz)
    nn.updateWeights(eta)



