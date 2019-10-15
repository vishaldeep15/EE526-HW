#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 20:04:08 2019

@author: vishal
"""

import numpy as np 
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import time

# Get data from tensor flow
mnist = tf.keras.datasets.mnist
# Split data between train and test data
(x, y),(x_test, y_test) = mnist.load_data()
# flatten the image data for all 60000 images
X = x.reshape(-1, 784)
Xtest = x_test.reshape(-1, 784)
# One hot encoding
y = to_categorical(y)
ytest = to_categorical(y_test)

print(X.shape)
print(y.shape)

model = Sequential([
  Dense(50, activation='relu', input_shape=(784,)),
  Dense(50, activation='relu'),
  Dense(10, activation='softmax'),
])

sgd = optimizers.SGD(lr=0.5)

model.compile(
  optimizer=sgd,
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

model.fit(
  X, # training data
  y, # training targets
  epochs=10,
  batch_size=500
)

#model.evaluate(
#  Xtest,
#  ytest
#)

[loss, accuracy] = model.evaluate(x=Xtest, y=ytest, batch_size=500, verbose=1)

print(loss)
print(accuracy)

