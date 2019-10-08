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

  # Problem 4 part A)
  aLayers = [(10, Linear)]
  aCE = ObjectiveFunction('crossEntropyLogit')
  aNN = NeuralNetwork(nInputs, aLayers, M)

  # Problem 4 part B)
  layers=[(50, ReLU), (50, ReLU), (10, Linear)]
  CE=ObjectiveFunction('crossEntropyLogit')
  nn=NeuralNetwork(nInputs, layers, M)

  # Problem 4 part C)
  cLayers = [(140, ReLU), (10, Linear)]
  cCE = ObjectiveFunction('crossEntropyLogit')
  cNN = NeuralNetwork(nInputs, cLayers, M)

  nIter=10000
  B=100
  eta=0.1 # learning rate
  for i in range(nIter):
    x, y = mnist.next_batch(B)

    # Part A
    alogit = aNN.doForward(x)
    aJ = aCE.doForward(alogit, y)
    adp = aCE.doBackward(y)
    aNN.doBackward(adp)
    aNN.updateWeights(eta)

    # Part B
    logit = nn.doForward(x)
    J=CE.doForward(logit, y)
    dp=CE.doBackward(y)
    nn.doBackward(dp)
    nn.updateWeights(eta)

    # Part C
    clogit = cNN.doForward(x)
    cJ = cCE.doForward(clogit, y)
    cdp = cCE.doBackward(y)
    cNN.doBackward(cdp)
    cNN.updateWeights(eta)

    # compute error rate
    if (i==nIter-1):  # i%100==0 or
      trainPred = alogit.argmax(axis=0)
      trainActual = y.argmax(axis=0)
      trainAccu = sum(trainPred == trainActual) / len(trainActual)
      print("Part A)")
      print("training error: iter %d, J=%f, accu=%.2f" % (i, aJ, trainAccu))

      p = aNN.doForward(mnist.x_test)
      yhat = p.argmax(axis=0)
      yTrue = mnist.y_test.argmax(axis=0)
      accu = sum(yhat == yTrue) / len(yTrue)
      print('\rtest error: iter %d, J=%f, accu=%.2f' % (i, aJ, accu))

      # part b
      trainPred = logit.argmax(axis=0)
      trainAccu = sum(trainPred==trainActual)/len(trainActual)
      print("\nPart B)")
      print("training error: iter %d, J=%f, accu=%.2f"% (i, J, trainAccu))

      p = nn.doForward(mnist.x_test)
      yhat = p.argmax(axis=0)
      accu = sum(yhat == yTrue) / len(yTrue)
      print('\rtest error: iter %d, J=%f, accu=%.2f' % (i, J, accu))

      # part c
      trainPred = clogit.argmax(axis=0)
      trainAccu = sum(trainPred == trainActual) / len(trainActual)
      print("\nPart C)")
      print("training error: iter %d, J=%f, accu=%.2f" % (i, cJ, trainAccu))

      p=cNN.doForward(mnist.x_test)
      yhat=p.argmax(axis=0)
      accu = sum(yhat==yTrue)/len(yTrue)
      print( '\rtest error: iter %d, J=%f, accu=%.2f' % (i, cJ, accu))

main()
# run: python %
