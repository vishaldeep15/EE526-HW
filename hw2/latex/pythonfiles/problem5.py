# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:03:55 2019
Problem: 5, Homework 2
@author: vishal Deep
"""
import time
from DNN import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical

def trainAndTestSplit(spamEmails, notSpamEmails):
    # convert data into numpy arrays
    spamEmailsArr = spamEmails.to_numpy()
    notSpamEmailsArr = notSpamEmails.to_numpy()    
    # Calculate number of total spam and not spam emails
    totalSpam = spamEmails.shape[0]
    totalNotSpam = notSpamEmails.shape[0]    
    # Calculate number of spam and not spam train and test
    numSpamTrain = int(np.floor((2/3) * totalSpam))
    numNotSpamTrain = int(np.floor((2/3) * totalNotSpam))
    numSpamTest = totalSpam - numSpamTrain
    numNotSpamTest = totalNotSpam - numNotSpamTrain 
    # Seperate spam and not spam train and test arrays
    trainSpam = spamEmailsArr[0: numSpamTrain, :]
    trainNotSpam = notSpamEmailsArr[0: numNotSpamTrain, :]
    testSpam = spamEmailsArr[numSpamTrain:numSpamTrain+numSpamTest, :]
    testNotSpam = notSpamEmailsArr[numNotSpamTrain:numNotSpamTrain+numNotSpamTest, :]
    # Combine test and train data
    trainData = np.vstack((trainSpam, trainNotSpam))
    testData = np.vstack((testSpam, testNotSpam))
    return trainData, testData

def generateLabel(y_pred):
    y_pred[y_pred < 0] = 0
    y_pred[y_pred > 0] = 1
    return np.transpose(y_pred)

# read data from the file
spamBaseData = pd.read_csv("spambase/spambase.data", header=None)
# Extract label y 
y = spamBaseData.iloc[:, -1]
# filter spam
spamEmails = spamBaseData.loc[y == 1]
# filtet not spam
notSpamEmails = spamBaseData.loc[y == 0]
# split the train and test data
[trainData, testData] = trainAndTestSplit(spamEmails, notSpamEmails)
# seperate X and y of training and test data
Xtrain = trainData[:, 0:-1]
ytrain = trainData[:, -1]
Xtest = testData[:, 0:-1]
ytest = testData[:, -1]

X = np.transpose(Xtrain)
y = np.transpose(ytrain)
y = y.reshape(1,3066)

D = 57 # Input dimension
Odim = 1 # number of outputs
layers=[ (100, ReLU), (40, ReLU), (Odim, Linear) ]    
# initialize Neural Network with D inputs and layers
nn = NeuralNetwork(D, layers)
# set random weights with maximum size of 0.1
nn.setRandomWeights(0.1)
# select crossentropy as objective function
CE = ObjectiveFunction('logistic')
eta = 1e-1

startTime = np.round(time.time(), decimals=4)
for i in range(10000):
    p = nn.doForward(X)
#    print(logp.shape)
    J    = CE.doForward(p, y)
    dz   = CE.doBackward(y)
    dx   = nn.doBackward(dz)
    nn.updateWeights(eta)
    if (i%100==0):
      print( '\riter %d, J=%f' % (i, J), end='')
stopTime = np.round(time.time(), decimals=4)
totalTime = np.round((stopTime - startTime), decimals=4)
print(f'\nTraining time: {totalTime} seconds')

# Prediction accuracy
X_test = np.transpose(Xtest)
p = nn.doForward(X_test)
y_pred = generateLabel(p)
# Performance calculations
perfArr = np.equal(y_pred, ytest)
accuracy = (np.sum(perfArr)/np.size(perfArr)) * 100
print(f'Accuracy: {accuracy}')