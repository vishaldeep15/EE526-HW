# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 20:46:24 2019

@author: visha
"""

import numpy as np
from keras.utils import to_categorical

def softmax(x):
    """Return the softmax of a vector x.
    
    :type x: ndarray
    :param x: vector input
    
    :returns: ndarray of same length as x
    """
#    x = x - np.max(x)
    row_sum = np.sum(np.exp(x))
    return np.array([np.exp(x_i) / row_sum for x_i in x])

def cross_entropy(y, s):
    """Return the cross-entropy of vectors y and s.

    :type y: ndarray
    :param y: one-hot vector encoding correct class

    :type s: ndarray
    :param s: softmax vector

    :returns: scalar cost
    """
    # Naively computes log(s_i) even when y_i = 0
    # return -y.dot(np.log(s))
    
    # Efficient, but assumes y is one-hot
    return -np.log(s[np.where(y)])

x = np.array([[-0.75,0.25],[0.25,-0.75]])
test = np.array([[1,2,3],[4,5,6]])

z = np.array([[0.25,-0.75],[-0.75,0.25]])
i2 = np.array([[1,0],[0,1]])

data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
# one hot encode
encoded = to_categorical(data)
print(encoded)

#print(test)
#print(np.transpose(test))

#print(z)
#print(softmax(z))
#print(np.matmul(x, i2))
#dz = softmax(z)-i2
colvec = np.array([[1],[1]])
#print(np.matmul(x, colvec))

#a = np.array([[0.5],[0.5]])
#print(a)
#b = np.array([[1, 1]])
#print(b)
#c = np.multiply(a, b)
#print(c)
#print(x-c)
dz1 = np.array([[-0.634,0.134],[0.134,-0.634]])
#print(np.matmul(dz1,colvec))




