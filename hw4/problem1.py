#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 02:53:38 2019

@author: vishal
"""

import tensorflow as tf
#import numpy as np
import matplotlib.pyplot as plt

# Load fashion MNIST data set
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

print("x_train shape:", x_test.shape, "y_train shape:", y_test.shape)

plt.imshow(x_train[500])

# Normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

print(x_train[0:10])