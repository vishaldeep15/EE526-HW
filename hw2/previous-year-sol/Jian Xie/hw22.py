
# coding: utf-8

# In[ ]:


import numpy as np
class crossEntropyLogit:

  
  def doForward(self,z,y):
    aa=z.max(axis=0)
    logp=(z-aa)-np.log(np.sum(np.exp(z-aa),axis=0))
    J=-np.sum(logp*y)/y.shape[1]
    self.logp=logp
    return J
  def doBackward(self,y):
    dz=np.exp(self.logp)-y
    m=y.shape[1]
    dz*=1/m
    return dz

CE=crossEntropyLogit()
np.random.seed(1)
z=np.random.rand(3,2)
y=np.eye(3)[:,:2]
J1=CE.doForward(z,y)
J2=CE.doForward(z+1000,y)
dz=CE.doBackward(y)
print(J1)
print(J2)
print(dz)

