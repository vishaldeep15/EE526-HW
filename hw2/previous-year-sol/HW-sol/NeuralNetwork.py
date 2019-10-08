# Multi-Layer Neural Network Implementation
# Zhengdao Wang

import numpy as np

class Linear:
  def forward(self, x): return x
  def backward(self, dy): return dy

class ReLU:
  ''' ReLU nonlinearity '''
  def forward(self, x):
    y=np.maximum(0, x)
    self.x=x
    return y

  def backward(self, dy):
    dx = (self.x>0)*dy
    return dx

class Layer:
  ''' One neural layer -- using ReLU nonlinearity'''

  def __init__(self, nInputs, nOutputs, NL):
    ''' nInputs : number of inputs
        nOutputs: number of outputs
        NL : ReLU, Logistic, or softmax objects
    '''
    self.nInputs=nInputs
    self.nOutputs=nOutputs
    self.W=np.zeros((nOutputs, nInputs))
    self.b=np.zeros((nOutputs, 1))
    self.dW=np.zeros((nOutputs, nInputs))
    self.db=np.zeros((nOutputs, 1))
    self.NL=NL()

  def setRandomWeights(self, M=0.1):
    ''' set random uniform weights of max size M '''
    self.W=np.random.rand(self.nOutputs, self.nInputs)*M
    self.b=np.random.rand(self.nOutputs, 1)*M

  def doForward(self, _input):
    ''' Given the input to the Layer, return the output
    Also store the necessary values for backward propagation '''
    ....

  def doBackward(self, dOutput):
    '''Given the gradient of the output, return the gradient of the input.
    Also store the gradient with respect to W and b'''
    ....

  def updateWeights(self, eta):
    ''' Gradient descent update of W and b, using the gradient stored in
    W and b'''
    ....

class NeuralNetwork:
  ''' a neural network '''

  def __init__(self, nInputs, layers, M=1e-1):
    ''' layers=[ (n_neurons, NL), ... ]
    Note that A[l] will be the input to layer l, and A[l+1] will be
    the output of layer l, l=0, 1, ..., L-1, where L is the total number
    of neural layers (not counting the input, which has no neurons).
    The index of A is "one off" from that of notes, because Python is
    zero-based, and we number all variables from 0.
    '''
    self.nLayers=len(layers)
    self.A=[None]*(self.nLayers+1)  # input + all activations
    self.dA=[None]*(self.nLayers+1) # d(input + all activations)
    self.layers=[ Layer(nInputs, layers[0][0], layers[0][1]) ]
    for l in range(1,self.nLayers):
      self.layers.append( Layer(layers[l-1][0], layers[l][0], layers[l][1]) )
    self.setRandomWeights(M)

  def setRandomWeights(self, M=1e-1):
    for l in range(self.nLayers):
      self.layers[l].setRandomWeights(M)

  def doForward(self, _input):
    ''' Given the network input, return the network output. And store all
    intermediate results (including network input and output) in A.'''
    ....

  def doBackward(self, dOutput):
    ''' Given the network output gradient, return the network input
    gradient. Store all intermediate gradients (including gradients for
    network input and output) in dA.'''
    ....

  def updateWeights(self, eta):
    for l in range(self.nLayers):
      self.layers[l].updateWeights(eta)

  def print(self, want=['W', 'dW', 'b', 'db', 'Z', 'dZ', 'Input'
        'Output', 'dInput', 'dOutput']):
    for l in range(self.nLayers):
      Map={'W':self.layers[l].W,
        'dW': self.layers[l].dW,
        'b': self.layers[l].b,
        'db': self.layers[l].db,
        'Z ': self.layers[l].Z,
        'dZ': self.layers[l].dZ,
        'Input': self.A[l],
        'Output': self.A[l+1],
        'dInput': self.dA[l],
        'dOutput': self.dA[l+1]}
      print('======== Layer %d =========' % l)
      for k in want:
        print('                           '+k + ':\n', Map[k])

class ObjectiveFunction:
  def crossEntropyLogitForward(self, logits, y):
    ''' Cross entropy between [log(p_1), ..., log(p_n)]+C, and
        [y_1, ..., y_n]. The former vector is called logits in Tensorflow.
        It can be obtained by just W*x+b, without any nonlinearity.
        Input logits and y are both n by m, where n is the number of classes
        and m is the number of data points.
    '''
    ....

  def crossEntropyLogitBackward(self, y):
    ''' Given the true labels, return the gradient with respect to
      the logits input (namely, y_hat), using stored self.logp '''
    ....

  def logisticForward(self, logits, y):
    ''' logit is w*x+b, one scalar for each data point.
        Input logits is 1 by m, where m is the number of points.
        Input y is 2 by m, first row y_0, and second row y_1.
    '''
    logits=np.vstack( (np.zeros((1, logits.size)), logits) ) # padding 0 on top
    return self.crossEntropyLogitForward(logits, y)

  def logisticBackward(self, y):
    dz=self.crossEntropyLogitBackward( np.vstack( (1-y,y) ) )
    return dz[1,None]

  def __init__(self, name):
    if name=='crossEntropyLogit':
      self.doForward=self.crossEntropyLogitForward
      self.doBackward=self.crossEntropyLogitBackward
    elif name=='logistic':
      self.doForward=self.logisticForward
      self.doBackward=self.logisticBackward
