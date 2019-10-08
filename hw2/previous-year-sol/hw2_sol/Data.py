#!/usr/bin/python

# custom data package

import numpy as np

class Data:
  ''' data points always indexed by columns (each column a point)
    Date((x,y), (x_test,y_test)) -- get training and testing data
    Date((x,y)) -- get both training and testing data together
                   A further split is needed
  '''

  def __init__(self, training, testing=None):
    self.x,self.y=np.copy(training[0]),np.copy(training[1])
    # make sure a matrix
    if self.y.ndim==1: self.y=self.y.reshape(1, self.y.size)
    if testing != None:
      self.x_test,self.y_test=np.copy(testing[0]),np.copy(testing[1])
      if self.y_test.ndim==1:
        self.y_test=self.y_test.reshape(1, self.y_test.size)
    self.N=self.y.shape[1]
    self.start=0 # for next_batch

  def split(self, ratio=0.8, shuffle=True):
    ''' ratio is the percentage of training samples '''
    if self.x_test != None: # recombine data first
      x=np.hstack( (self.x, self.x_test) )
      y=np.hstack( (self.y, self.y_test) )
    else:
      x,y=self.x,self.y
    M=int( np.floor(ratio*self.x.shape[1]) );
    if shuffle==True:
      x=np.random.shuffle(x.T).T
      y=np.random.shuffle(y.T).T
    self.x,self.y=x[:,:M],y[:,:M]
    self.x_test,self.y_test=x[:,M:],y[:,M:]

  @classmethod
  def fromName(cls, name): # based on names
    if name=='MNIST':
      import tensorflow as tf
      mnist = tf.keras.datasets.mnist
      (x,y),(x_test,y_test)=mnist.load_data()
      x=x.transpose((1,2,0)).reshape(784,-1,order='F')
      x_test=x_test.transpose((1,2,0)).reshape(784,-1, order='F')
      return cls( (x, y), (x_test, y_test) )

  def normalize(self):
    ''' normalize rows of X so that each row has mean zero and variance 1'''
    X=np.hstack( (self.x, self.x_test) )
    m=np.mean(X, axis=1, keepdims=True)
    var=np.var(X, axis=1, keepdims=True)
    var[var<1E-5]=1
    self.x=self.x-m; self.x_test=self.x_test-m;
    self.x=self.x/np.sqrt(var); self.x_test=self.x_test/np.sqrt(var)

  def toOneHot(self):
    if self.y.shape[0]!=1:
      raise ValueError('cannot do onehot encoding of non-scalars')
    def onehot(a):
      b = np.zeros((a.max()+1, a.size))
      b[a, np.arange(a.size)] = 1
      return b
    self.y=onehot(self.y[0])
    if self.y_test is not None:
      self.y_test=onehot(self.y_test[0])

  def next_batch(self, num, shuffle=False):
    if shuffle==True:
      idx = np.arange(0,self.N)
      np.random.shuffle(idx)
      idx = idx[:num]
      x=self.x[:,idx]
      y=self.y[:,idx]
      return x,y
    else:
      idx=np.arange(self.start, self.start+num)
      idx=np.mod(idx, self.N)
      x=self.x[:,idx]
      y=self.y[:,idx]
      self.start+=num
      return x,y
