# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:03:55 2019
Problem: 6, Homework 2
@author: vishal Deep
"""
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from DNN import *
from keras.utils import to_categorical
from sklearn import preprocessing
import time

# Get data from tensor flow
mnist = tf.keras.datasets.mnist
# Split data between train and test data
(x, y),(x_test, y_test) = mnist.load_data()
# flatten the image data for all 60000 images
X = x.transpose((1, 2, 0)).reshape(784, -1)
y = np.transpose(to_categorical(y))
# Scale data with mean=0 and var=1
X_scaled = preprocessing.scale(X)

D = 784 # Input dimension
Odim = 10 # number of outputs
layers=[(50, ReLU), (50, ReLU), (Odim, Linear)]
# initialize Neural Network with D inputs and layers
nn = NeuralNetwork(D, layers)
# set random weights with maximum size of 0.1
nn.setRandomWeights(0.1)
# select crossentropy as objective function
CE = ObjectiveFunction('crossEntropyLogit')

eta = 1e-1

startTime = np.round(time.time(), decimals=4)
# Train with training data
for i in range(100000):
    logp = nn.doForward(X_scaled)
    J    = CE.doForward(logp, y)
    dz   = CE.doBackward(y)
    dx   = nn.doBackward(dz)
    nn.updateWeights(eta)
    if (i%100==0):
      print( '\riter %d, J=%f' % (i, J), end='')
stopTime = np.round(time.time(), decimals=4)
totalTime = (np.round((stopTime - startTime), decimals=4))/60
print(f'\nTraining time: {totalTime} minutes')

# Prediction accuracy
X_test = np.transpose(x_test)
X_test = X_test.transpose((1, 2, 0)).reshape(784, -1)
X_test_scaled = preprocessing.scale(X_test)

p = nn.doForward(X_test_scaled)
yhat=p.argmax(axis=0)
accu = sum(yhat==y_test)/len(y_test)
# Performance calculations
perfArr = np.equal(yhat, y_test)
accuracy = (np.sum(perfArr)/np.size(perfArr)) * 100
print(f'Accuracy: {accuracy}')