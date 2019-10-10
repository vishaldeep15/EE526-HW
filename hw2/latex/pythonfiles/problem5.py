# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:03:55 2019
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

def calcAccuracy(p, y):
    y_pred = generateLabel(p)
    perfArr = np.equal(y_pred, y)
    accuracy = (np.sum(perfArr)/np.size(perfArr)) * 100
    return accuracy

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

y = ytrain.reshape(1, 3066)

D = 57 # Input dimension
Odim = 1 # number of outputs
layers=[ (100, ReLU), (40, ReLU), (Odim, Linear) ]    
# initialize Neural Network with D inputs and layers
nn = NeuralNetwork(D, layers)
# select crossentropy as objective function
CE = ObjectiveFunction('logistic')
eta = [0.01, 0.1, 0.5, 1]

for eta in eta:
    # set random weights with maximum size of 0.1
    nn.setRandomWeights(0.1)
    # Print value of eta being used    
    print(f"Learning rate: {eta}\n")    
    # Record start time 
    startTime = np.round(time.time(), decimals=4)
    for i in range(10000):
        p = nn.doForward(X)
        J    = CE.doForward(p, y)
        dz   = CE.doBackward(y)
        dx   = nn.doBackward(dz)
        nn.updateWeights(eta)
        if (i%2000==0):
            accuracy = np.round(calcAccuracy(p, y), decimals=4)
            J_rounded = np.round(J, decimals=4)
            print(f'Iterations: {i}, J={J_rounded}, Training Accuracy: {accuracy} % \n') 
    
    # Calculate time taken to train      
    stopTime = np.round(time.time(), decimals=4)
    totalTime = (np.round(((stopTime - startTime)/60), decimals=4))
    print(f'Training time: {totalTime} minutes \n')
    
    # Test dataset Prediction accuracy
    X_test = np.transpose(Xtest)
    p = nn.doForward(X_test)
    accuracy = np.round(calcAccuracy(p, ytest), decimals=4)
    print(f'Test dataset accuracy: {accuracy} % \n')