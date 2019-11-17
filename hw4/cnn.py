#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 02:53:38 2019

@author: vishal
"""


import tensorflow as tf
from sklearn.model_selection import train_test_split
from time import time
import keras.backend as K
from tensorflow.python.keras.callbacks import TensorBoard

def processData():
    # Load fashion MNIST data set
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    # Split into train and validation data
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)
    # reshape to conform with size 
    x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
    x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
    x_val = x_val.reshape((x_val.shape[0], 28, 28, 1))
    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_val = x_val.astype('float32') / 255
    # Do one hot encoding
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return x_train, y_train, x_val, y_val, x_test, y_test

# The network is adapted from the
def createModel():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=(28, 28, 1))) 
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.summary()
    return model

def getFlops(model):
    run_meta = tf.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    # Keras session graph in the call to the profiler.
    flops = tf.compat.v1.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
    return flops.total_float_ops

batch_size = 64
epochs = 10
(x_train, y_train, x_val, y_val, x_test, y_test) = processData()

print("x_train shape:", x_train.shape, "y_train shape:", y_train.shape)

model = createModel()

# Print number of flops 
print(getFlops(model))

model.compile(loss='categorical_crossentropy', 
             optimizer='adam',
             metrics=['accuracy'])

tensoboard = TensorBoard(log_dir="logs/{}".format(time()))

# Train the network
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[tensoboard])

# Evaluate the model on test data
[loss, acc] = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', acc)