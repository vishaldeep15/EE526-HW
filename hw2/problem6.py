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

testFlag = True
# Get data from tensor flow
mnist = tf.keras.datasets.mnist
# Split data between train and test data
(x, y),(x_test, y_test) = mnist.load_data()
# flatten the image data for all 60000 images
X = x.transpose((1, 2, 0)).reshape(784, -1)
y = np.transpose(to_categorical(y))
# Scale data with mean=0 and var=1
X_scaled = preprocessing.scale(X)
if testFlag:
    X_scaled = X_scaled[:, 0:1000]
    y = y[:, 0:1000]
print(X_scaled.shape)

D = 784 # Input dimension
Odim = 10 # number of outputs

layers_a = [(Odim, Linear)]
layers_b = [(50, ReLU), (50, ReLU), (Odim, Linear)]
layers_c = [(100, ReLU), (50, ReLU), (Odim, Linear)]

# initialize Neural Network with D inputs and layers
nn_a = NeuralNetwork(D, layers_a)
nn_b = NeuralNetwork(D, layers_b)
nn_c = NeuralNetwork(D, layers_c)

# set random weights with maximum size of 0.1
nn_a.setRandomWeights(0.1)
nn_b.setRandomWeights(0.1)
nn_c.setRandomWeights(0.1)

# select crossentropy as objective function
CE_a = ObjectiveFunction('crossEntropyLogit')
CE_b = ObjectiveFunction('crossEntropyLogit')
CE_c = ObjectiveFunction('crossEntropyLogit')

eta = 1e-1

startTime = np.round(time.time(), decimals=4)
# Train network a with training data
for i in range(10000):
    logp_a = nn_a.doForward(X_scaled)
    J_a    = CE_a.doForward(logp_a, y)
    dz_a   = CE_a.doBackward(y)
    dx_a   = nn_a.doBackward(dz_a)
    nn_a.updateWeights(eta)
    if (i%1000==0):
        yhat_a = logp_a.argmax(axis=0)
        perfArr = np.equal(yhat_a, y)
        accuracy_a = (np.sum(perfArr)/np.size(perfArr)) * 100
        J_a = np.round(J_a, decimals=4)
        print(f'Iterations: {i}, J={J_a}, Training Accuracy: {accuracy_a} \n')    
      
stopTime = np.round(time.time(), decimals=4)
totalTime = np.round(((stopTime - startTime)/60), decimals=4)
print(f'Training time NN_a: {totalTime} minutes')

# Train network b with training data
startTime = np.round(time.time(), decimals=4)
for i in range(10000):
    logp_b = nn_a.doForward(X_scaled)
    J_b    = CE_a.doForward(logp_b, y)
    dz_b   = CE_a.doBackward(y)
    dx_b   = nn_a.doBackward(dz_b)
    nn_b.updateWeights(eta)
    if (i%1000==0):
        yhat_b = logp_b.argmax(axis=0)
        perfArr = np.equal(yhat_b, y)
        accuracy_b = (np.sum(perfArr)/np.size(perfArr)) * 100
        J_b = np.round(J_b, decimals=4)
        print(f'Iterations: {i}, J={J_b}, Training Accuracy: {accuracy_b} \n')    
      
stopTime = np.round(time.time(), decimals=4)
totalTime = np.round(((stopTime - startTime)/60), decimals=4)
print(f'Training time NN_b: {totalTime} minutes')

# Train network c with training data
startTime = np.round(time.time(), decimals=4)
for i in range(10000):
    logp_c = nn_c.doForward(X_scaled)
    J_c    = CE_c.doForward(logp_c, y)
    dz_c   = CE_c.doBackward(y)
    dx_c   = nn_c.doBackward(dz_c)
    nn_c.updateWeights(eta)
    if (i%1000==0):
        yhat_c = logp_c.argmax(axis=0)
        perfArr = np.equal(yhat_c, y)
        accuracy_c = (np.sum(perfArr)/np.size(perfArr)) * 100
        J_c = np.round(J_c, decimals=4)
        print(f'Iterations: {i}, J={J_c}, Training Accuracy: {accuracy_c} \n')    
      
stopTime = np.round(time.time(), decimals=4)
totalTime = np.round(((stopTime - startTime)/60), decimals=4)
print(f'Training time NN_c: {totalTime} minutes')

# Prediction accuracy
#X_test = np.transpose(x_test)
#X_test = X_test.transpose((1, 2, 0)).reshape(784, -1)
#X_test_scaled = preprocessing.scale(X_test)
#
#p = nn.doForward(X_test_scaled)
#yhat=p.argmax(axis=0)
#accu = sum(yhat==y_test)/len(y_test)
## Performance calculations
#perfArr = np.equal(yhat, y_test)
#accuracy = (np.sum(perfArr)/np.size(perfArr)) * 100
#print(f'Accuracy: {accuracy}')