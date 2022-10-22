from cgi import test
import sys
import os
import numpy as np
import pandas as pd
import math

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)
NUM_FEATS = 90

class Net(object):
    def __init__(self, num_layers, num_units):
        self.num_layers = num_layers
        self.num_units = num_units
        self.biases = []
        self.weights = []
        for i in range(num_layers):
            if i==0:
				# Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units)))
            else:
				# Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, self.num_units)))
            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))

		# Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(4, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 4)))
		#print(self.biases)
    def  activation(self,h):
        return np.maximum(0,h)    
    def softmax(self,h):
        h=np.exp(h)
        h=h/h.sum(axis=1,keepdims=True)
        return h
    def __call__(self, X):
        '''
		Forward propagate the input X through the network,
		and return the output.

		Note that for a classification task, the output layer should
		be a softmax layer. So perform the computations accordingly

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
		Returns
		----------
			y : Output of the network, numpy array of shape m x 1
		'''
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = np.dot(a, w) + b.T
            if i < len(self.weights)-1:
                a = self.activation(h)
            else: 
                output_layer = self.softmax(h)
        return output_layer
    def forward(self,X)	:
        a = X
        activations=[]
        activations.append(X)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = np.dot(a, w) + b.T
            if i < len(self.weights)-1:
                a = self.activation(h)
                activations.append(a)
            else: # No activation for the output layer
               a=self.softmax(h)

        return [activations,a]
    def backward(self, X, y, lamda):
        '''
		Compute and return gradients loss with respect to weights and biases.
		(dL/dW and dL/db)

		Parameters
		----------
			X : Input to the network, numpy array of shape m x d
			y : Output of the network, numpy array of shape m x 1
			lamda : Regularization parameter.

		Returns
		----------
			del_W : derivative of loss w.r.t. all weight values (a list of matrices).
			del_b : derivative of loss w.r.t. all bias values (a list of vectors).

		Hint: You need to do a forward pass before performing backward pass.
		'''
        answer,pred= self.forward(X) #forward pass
        batch_size=y.shape[0]
        delY= 
        del_W=[]
        del_B=[]
        for (w,a) in reversed(list(zip(self.weights,answer))):
            delW=np.dot(a.T,delY)+ (lamda*w)
            delB=np.sum(delY,axis=0)
            delB = np.reshape(delB,(len(delB),1))
            delX=np.dot(delY,w.T)
            delY=delX
            del_W.append(delW)
            del_B.append(delB)
            del_W.reverse()
            del_B.reverse()
        return del_W,del_B
    
  
