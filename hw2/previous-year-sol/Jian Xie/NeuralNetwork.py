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
    self.A=_input
    Z=self.W.dot(_input)+self.b
    _output=self.NL.forward(Z)
    return _output

  def doBackward(self, dOutput):
    '''Given the gradient of the output, return the gradient of the input.
    Also store the gradient with respect to W and b'''
    dZ=self.NL.backward(dOutput)
    self.dW=dZ@self.A.T
    self.db=np.sum(dZ,axis=1,keepdims=True)
    dInput=self.W.T@dZ
    return dInput

  def updateWeights(self, eta):
    ''' Gradient descent update of W and b, using the gradient stored in
    W and b'''
    self.W-=eta*self.dW
    self.b-=eta*self.db

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
    self.A[0]=_input 
    for l in range(self.nLayers):
        self.A[l+1]=self.layers[l].doForward(self.A[l])
    return self.A[self.nLayers]

  def doBackward(self, dOutput):
    ''' Given the network output gradient, return the network input
    gradient. Store all intermediate gradients (including gradients for
    network input and output) in dA.'''
    self.dA[self.nLayers]=dOutput
    for l in range(self.nLayers,0,-1):
        self.dA[l-1]=self.layers[l-1].doBackward(self.dA[l])
    #return self.dA[0]

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
    a=logits.max(axis=0)
    logp=(logits-a)-np.log(np.sum(np.exp(logits-a),axis=0))
    J=-np.sum(logp*y)/y.shape[1]
    self.logp=logp
    return J

  def crossEntropyLogitBackward(self, y):
    ''' Given the true labels, return the gradient with respect to
      the logits input (namely, y_hat), using stored self.logp '''
    dz=np.exp(self.logp)-y
    m=y.shape[1]
    dz*=1/m
    return dz

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
