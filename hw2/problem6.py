# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:03:55 2019
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

def calcAccuracy(logp, y):
    yhat = logp.argmax(axis=0)
    perfArr = np.equal(yhat, y)
    accuracy = (np.sum(perfArr)/np.size(perfArr)) * 100
    return accuracy

# create a list containing mini-batches 
def createMiniBatches(X, y, batch_size): 
    mini_batches = [] 
    X = np.transpose(X)
    y = y.reshape((-1, 1)) 
    data = np.hstack((X,y))
    np.random.shuffle(data)     
    n_minibatches = data.shape[0] // batch_size     
    for i in range(n_minibatches): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = np.transpose(mini_batch[:, :-1])
        Y_mini = mini_batch[:, -1]
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches 

testFlag = False
# Get data from tensor flow
mnist = tf.keras.datasets.mnist
# Split data between train and test data
(x, y),(x_test, y_test) = mnist.load_data()
# flatten the image data for all 60000 images
X = x.transpose((1, 2, 0)).reshape(784, -1)
# Normalize data
X_norm = np.divide(X, 255)

if testFlag:
    X_norm = X_norm[:, 0:10000]
    y = y[0:10000]

# Create mini batches of 100    
mini_batches = createMiniBatches(X_norm, y, 100)

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

numOfIterations = 10000
batchCount = 1
eta = 1e-2
print(f"Learning rate: {eta}\n")    

for batch in mini_batches:
    X = batch[0]
    y = batch[1]
    # One hot encoding
    y = np.transpose(to_categorical(y))

    # Train network a with training data
    startTime = np.round(time.time(), decimals=4)
    for i in range(numOfIterations):
        logp_a = nn_a.doForward(X)
        J_a    = CE_a.doForward(logp_a, y)
        dz_a   = CE_a.doBackward(y)
        dx_a   = nn_a.doBackward(dz_a)
        nn_a.updateWeights(eta)

    stopTime = np.round(time.time(), decimals=4)
    totalTime_a = np.round(((stopTime - startTime)/60), decimals=4)
#    print(f'Training time NN_a: {totalTime} minutes \n')
    
    # Train network b with training data
    startTime = np.round(time.time(), decimals=4)
    for i in range(numOfIterations):
        logp_b = nn_b.doForward(X)
        J_b    = CE_b.doForward(logp_b, y)
        dz_b   = CE_b.doBackward(y)
        dx_b   = nn_b.doBackward(dz_b)
        nn_b.updateWeights(eta)

    stopTime = np.round(time.time(), decimals=4)
    totalTime_b = np.round(((stopTime - startTime)/60), decimals=4)
#    print(f'Training time NN_b: {totalTime} minutes \n')
    
    # Train network c with training data
    startTime = np.round(time.time(), decimals=4)
    for i in range(numOfIterations):
        logp_c = nn_c.doForward(X)
        J_c    = CE_c.doForward(logp_c, y)
        dz_c   = CE_c.doBackward(y)
        dx_c   = nn_c.doBackward(dz_c)
        nn_c.updateWeights(eta)
       
    stopTime = np.round(time.time(), decimals=4)
    totalTime_c = np.round(((stopTime - startTime)/60), decimals=4)
#   print(f'Training time NN_c: {totalTime} minutes \n')
    
    if (batchCount%20 == 0):
        print(f"Batch Completed: {batchCount}\n")
        print(f'Training time NN_a: {totalTime_a} minutes \n')
        print(f'Training time NN_b: {totalTime_a} minutes \n')
        print(f'Training time NN_c: {totalTime_c} minutes \n')
        accuracy_a = np.round(calcAccuracy(logp_a, y), decimals=4)
        J_a = np.round(J_a, decimals=4)
        print(f'Iterations: {i}, J={J_a}, Training Accuracy: {accuracy_a} % \n')  
        accuracy_b = np.round(calcAccuracy(logp_b, y), decimals=4)
        J_b = np.round(J_b, decimals=4)
        print(f'Iterations: {i}, J={J_b}, Training Accuracy: {accuracy_b} % \n')
        accuracy_c = np.round(calcAccuracy(logp_c, y), decimals=4)
        J_c = np.round(J_c, decimals=4)
        print(f'Iterations: {i}, J={J_c}, Training Accuracy: {accuracy_c} % \n')
    
    batchCount += 1

# Test Data Prediction accuracy 
X_test = np.transpose(x_test)
X_test = X_test.transpose((1, 2, 0)).reshape(784, -1)
# Normalize data
X_test_norm = np.divide(X_test, 255)

p_a = nn_a.doForward(X_test_norm)
p_b = nn_b.doForward(X_test_norm)
p_c = nn_c.doForward(X_test_norm)

accuracy_a = calcAccuracy(p_a, y_test)
accuracy_b = calcAccuracy(p_b, y_test)
accuracy_c = calcAccuracy(p_c, y_test)
print(f'Test Accuracy NN_a: {accuracy_a}%')
print(f'Test Accuracy NN_b: {accuracy_b}%')
print(f'Test Accuracy NN_c: {accuracy_c}%')