#!/usr/bin/python

from Data import *
import numpy as np
from NeuralNetwork import *

def main():
  mnist=Data.fromName('MNIST')
  mnist.normalize()
  mnist.toOneHot()

  np.random.seed(1)

  M=1E-1  # initial weight size
  nInputs=784
 # layers=[(90, ReLU), (50, ReLU), (10, Linear)] # setting (c)
 # layers = [(10, Linear)]  # setting (a)
  layers=[(50, ReLU), (50, ReLU), (10, Linear)]#setting(b)
  CE=ObjectiveFunction('crossEntropyLogit')
  nn=NeuralNetwork(nInputs, layers, M)

  nIter=10000
  B=100
  eta=0.1 # learning rate
  for i in range(nIter):
    x, y = mnist.next_batch(B)
    logit = nn.doForward(x)
    J=CE.doForward(logit, y)#caculate J
    dp=CE.doBackward(y)#caculate dp
    nn.doBackward(dp)
    nn.updateWeights(eta)

    # compute error rate
    if (i%100==0):
      p=nn.doForward(mnist.x_test)
      yhat=p.argmax(axis=0)
      yTrue=mnist.y_test.argmax(axis=0)
      accu = 1-sum(yhat==yTrue)/len(yTrue)
      p2=nn.doForward(x)
      ytrain=p2.argmax(axis=0)
      ytraintrue=y.argmax(axis=0)
      accu2=1-sum(ytrain==ytraintrue)/len(ytrain)
      print( '\riter %d, J=%f, train error=%.2f,test error=%.2f' % (i, J, accu2 ,accu))

main()
# run: python %
