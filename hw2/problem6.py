import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf
from DNN import *

# Get data from tensor flow
mnist = tf.keras.datasets.mnist
# Split data between train and test data
(x, y),(x_text, y_test) = mnist.load_data()
# flatten the image data for all 60000 images
X = x.transpose((1, 2, 0)).reshape(784, -1)

# print(X.shape[1])
#print(y.shape)

D = 784 # Input dimension
Odim = 10 # number of outputs
layers=[(Odim, Linear)]

# initialize Neural Network with D inputs and layers
nn = NeuralNetwork(D, layers)
# set random weights with maximum size of 0.1
nn.setRandomWeights(0.1)

# select crossentropy as objective function
CE = ObjectiveFunction('crossEntropyLogit')

eta = 1e-1

print(X.shape)
#y.reshape(1, 60000)
print(y.shape)
print(y_test.shape)
print(y)

for i in range(10000):
    logp = nn.doForward(X)
    J    = CE.doForward(logp, y)
    dz   = CE.doBackward(y)
    dx   = nn.doBackward(dz)
    nn.updateWeights(eta)

