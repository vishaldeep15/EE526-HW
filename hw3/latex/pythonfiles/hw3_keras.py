#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:04:08 2019

@author: vishal
"""

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

# Get data from tensor flow
mnist = tf.keras.datasets.mnist
# Split data between train and test data
(x, y),(x_test, y_test) = mnist.load_data()
# flatten the image data for all 60000 images
XTrain = x.reshape(-1, 784)
XTest = x_test.reshape(-1, 784)
# One hot encoding
yTrain = to_categorical(y)
yTest = to_categorical(y_test)

# Set up learning rate, epochs, and batch size
learningRate = 0.01
batchSize = 500
epochsArr = [10, 50, 100]

for epochs in epochsArr:
    print("========== NN1 session starts ===========")
    model_nn1 = Sequential([
      Dense(10, activation='softmax', input_shape=(784,))
    ])
    
    # Stochastic Gradient Descent optimizer
    sgd = optimizers.SGD(lr=learningRate, clipnorm=1.)
    # use crossentropy
    model_nn1.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the NN
    model_nn1.fit(XTrain, yTrain, epochs=epochs, batch_size=batchSize, verbose=2)
    # Calculate accuracy on test data
    [loss_nn1, accuracy_nn1] = model_nn1.evaluate(x=XTest, y=yTest, batch_size=batchSize, verbose=2)
    print("Test Accuracy: {:f}".format(accuracy_nn1))
    print("========== NN1 session completed ===========")
    
    print("========== NN2 session starts ===========")
    model_nn2 = Sequential([
      Dense(50, activation='relu', input_shape=(784,)),
      Dense(50, activation='relu'),
      Dense(10, activation='softmax'),
    ])
    
    # Stochastic Gradient Descent optimizer
    sgd = optimizers.SGD(lr=learningRate, clipnorm=1.)
    # use crossentropy
    model_nn2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the NN
    model_nn2.fit(XTrain, yTrain, epochs=epochs, batch_size=batchSize, verbose=2)
    # Calculate accuracy on test data
    [loss_nn2, accuracy_nn2] = model_nn2.evaluate(x=XTest, y=yTest, batch_size=batchSize, verbose=2)
    print("Test Accuracy: {:f}".format(accuracy_nn2))
    print("========== NN2 session completed ===========")

