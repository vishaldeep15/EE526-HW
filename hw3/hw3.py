#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:20:36 2019

@author: vishal
"""

import numpy as np 
import tensorflow as tf
from keras.utils import to_categorical
from sklearn import preprocessing
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Return a batch of training data
def next_batch(batchSize, X, y):    
    idx = np.arange(0 , len(X))
    np.random.shuffle(idx)
    idx = idx[:batchSize]
    XBatch = X[idx]
    yBatch = y[idx]

    return XBatch, yBatch

# Python optimisation variables
learning_rate = 0.5
epochs = 100
batch_size =500

#Get data from tensor flow keras
#mnist = tf.keras.datasets.mnist
## Split data between train and test data
#(x, y),(x_test, y_test) = mnist.load_data()
## flatten the image data for all 60000 images
#X = x.transpose((0, 1, 2)).reshape(-1, 784)
## Normalize data
##X = np.divide(X, 255)
#y = to_categorical(y)
##y = np.transpose(to_categorical(y))
#
#total_batch = int(y.shape[0] / batch_size)
#
#print(X.shape)
#print(y)
#print(total_batch)


# Placeholders for X and y
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 50], stddev=0.01), name='W1')
b1 = tf.Variable(tf.random_normal([50]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([50, 50], stddev=0.01), name='W2')
b2 = tf.Variable(tf.random_normal([50]), name='b2')

W3 = tf.Variable(tf.random_normal([50, 10], stddev=0.01), name='W3')
b3 = tf.Variable(tf.random_normal([10]), name='b3')
# calculate the output of the hidden layer
hidden_out1 = tf.add(tf.matmul(x, W1), b1)
hidden_out1 = tf.nn.relu(hidden_out1)
hidden_out2 = tf.add(tf.matmul(hidden_out1, W2), b2)
hidden_out2 = tf.nn.relu(hidden_out2)


# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out2, W3), b3))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)  

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(mnist.train.labels) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
#            batch_x, batch_y = next_batch(batch_size, X, y)
#            print(batch_x.shape)
#            print(batch_y.shape)
            _, c = sess.run([optimiser, cross_entropy], 
                         feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))