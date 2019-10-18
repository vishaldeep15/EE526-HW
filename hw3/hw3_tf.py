#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 15:20:36 2019

@author: vishal
"""
import tensorflow as tf

# Import Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Set up learning rate, epochs, and batch size
learningRate = 0.01
epochsArr = [10, 50, 100]
batch_size = 500

total_batch = int(len(mnist.train.labels) / batch_size)

def layer(nInput, nOutput, X, activation):
    W = tf.Variable(tf.random_normal([nInput, nOutput], stddev=0.01))
    b = tf.Variable(tf.random_normal([nOutput]))
    z = tf.add(tf.matmul(X, W), b)
    if activation == 'ReLu':
        nOut = tf.nn.relu(z)
    elif activation == 'Linear':
        nOut = z
    else:
        raise ValueError('activation values can only be ReLu or Linear')
    return nOut

def calcAccuracy(logit, y):
    # find max value from y and logit and compare it
    accuratePredictions = tf.equal(tf.argmax(y, 1), tf.argmax(logit, 1))
    # Cast to a float32 and calculate mean
    accuracy = tf.reduce_mean(tf.cast(accuratePredictions, tf.float32))
    return accuracy

def logStat(nEpochs, epoch, avg_cost, accuracy):
    if nEpochs <= 10:
        if (epoch%2 == 0):
            print("Epoch: {}".format(epoch+1), "Cost: {:f}".format(avg_cost), "Training Acc: {:f}".format(accuracy))
    elif nEpochs > 10 and epochs <= 50:
        if (epoch%10 == 0):
            print("Epoch: {}".format(epoch+1), "Cost: {:f}".format(avg_cost), "Training Acc: {:f}".format(accuracy))
    elif nEpochs > 50 and nEpochs <= 100:
        if (epoch%20 == 0):
            print("Epoch: {}".format(epoch+1), "Cost: {:f}".format(avg_cost), "Training Acc: {:f}".format(accuracy))
    else:
        if (epoch%100 == 0):
            print("Epoch: {}".format(epoch+1), "Cost: {:f}".format(avg_cost), "Training Acc: {:f}".format(accuracy))

# Placeholders for X and y
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

for epochs in epochsArr:
    print("Number of epochs: {}".format(epochs))
    # First Neural Network
    # Layer 1
    l1_out_nn1 = layer(784, 10, x, 'Linear')
    # Softmax + cross Entropy
    ce_nn1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l1_out_nn1, labels=y))
    # Optimizer Gradient Descent
    optimizer_nn1 = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(ce_nn1)
    # Initialize variables
    init = tf.global_variables_initializer()
    # calculate accuracy
    accuracy_nn1 = calcAccuracy(l1_out_nn1, y)
    
    print("========== NN1 session starts ==========")
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init)
       
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                XBatch, yBatch = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimizer_nn1, ce_nn1], feed_dict={x: XBatch, y: yBatch})
                avg_cost += c / total_batch
            trainAcc_nn1 = sess.run(accuracy_nn1, feed_dict={x: XBatch, y: yBatch})
            logStat(epochs, epoch, avg_cost, trainAcc_nn1)
        testAcc_nn1 = sess.run(accuracy_nn1, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test Accuracy: {:f}".format(testAcc_nn1))
    
    print("========== NN1 session Completed ==========")
    
    # Second Neural Network
    l1_out_nn2 = layer(784, 50, x, 'ReLu')
    l2_out_nn2 = layer(50, 50, l1_out_nn2, 'ReLu')
    l3_out_nn2 = layer(50, 10, l2_out_nn2, 'Linear')
    
    # Softmax + cross Entropy
    ce_nn2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l3_out_nn2, labels=y))
    # Optimizer Gradient Descent
    optimizer_nn2 = tf.train.GradientDescentOptimizer(learning_rate=learningRate).minimize(ce_nn2)
    # finally setup the initialisation operator
    init = tf.global_variables_initializer()
    # calculate accuracy
    accuracy_nn2 = calcAccuracy(l3_out_nn2, y)
    
    print("========== NN2 session starts ==========")
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init)
       
        for epoch in range(epochs):
            avg_cost = 0
            for i in range(total_batch):
                XBatch, yBatch = mnist.train.next_batch(batch_size=batch_size)
                _, c = sess.run([optimizer_nn2, ce_nn2], feed_dict={x: XBatch, y: yBatch})
                avg_cost += c / total_batch
            trainAcc_nn2 = sess.run(accuracy_nn2, feed_dict={x: XBatch, y: yBatch})
#            print("Epoch: {}".format(epoch+1), "Cost: {:f}".format(avg_cost), "Training Acc: {:f}".format(trainAcc_nn2))
            logStat(epochs, epoch, avg_cost, trainAcc_nn2)    
        
        testAcc_nn2 = sess.run(accuracy_nn2, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test Accuracy: {:f}".format(testAcc_nn2))
    
    print("========== NN2 session Completed ==========")