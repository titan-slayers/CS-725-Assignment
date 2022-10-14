from cgi import test
import sys
import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt


# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90

class Net(object):
	'''
	'''

	def __init__(self, num_layers, num_units):
		'''
		Initialize the neural network.
		Create weights and biases.

		Here, we have provided an example structure for the weights and biases.
		It is a list of weight and bias matrices, in which, the
		dimensions of weights and biases are (assuming 1 input layer, 2 hidden layers, and 1 output layer):
		weights: [(NUM_FEATS, num_units), (num_units, num_units), (num_units, num_units), (num_units, 1)]
		biases: [(num_units, 1), (num_units, 1), (num_units, 1), (num_units, 1)]

		Please note that this is just an example.
		You are free to modify or entirely ignore this initialization as per your need.
		Also you can add more state-tracking variables that might be useful to compute
		the gradients efficiently.


		Parameters
		----------
			num_layers : Number of HIDDEN layers.
			num_units : Number of units in each Hidden layer.
		'''
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
		self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
		self.weights.append(np.random.uniform(-1, 1, size=(self.num_units, 1)))
		#print(self.biases)


	def activation(self,h):
		return np.maximum(0,h)

			
	


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
			else: # No activation for the output layer
				a=h
		return a
				

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
				a=h
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
		delY= 2./batch_size*(pred-y)
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



class Optimizer(object):
	'''
	Implementing Adam optimizer 

	Vt <- (beta * Vt-1) + (1-beta) *Gt
	St <- (gamma * St-1) + (1-gamma) *(Gt**2)
	Vtcap <- Vt/(1-beta^t)
	Stcap <- St/(1-gamma^t)

	Update rule 
	wt+1 <- wt - ((learning_rate*Vtcap)/root(Stcap+eps))

	Ideal values:
	beta : 0.9
	gamma : 0.999
	eps : 1e-8
	'''

	def __init__(self, learning_rate,B=0.9,Y=0.999,epsilon=1e-8):
		'''
		Create a Gradient Descent based optimizer with given
		learning rate.

		Other parameters can also be passed to create different types of
		optimizers.

		Hint: You can use the class members to track various states of the
		optimizer.
		'''
		
		self.B=B
		self.Y=Y		
		self.epsilon=epsilon
		self.learning_rate=learning_rate	
		self.v_dw=[]
		self.s_dw=[]
		self.v_db=[]
		self.s_db=[]
		self.t=0


	def step(self, weights, biases, delta_weights, delta_biases):
		'''
		Parameters
		----------
			weights: Current weights of the network.
			biases: Current biases of the network.
			delta_weights: Gradients of weights with respect to loss.
			delta_biases: Gradients of biases with respect to loss.

		'''
		
		
		if(self.t==0):
			for (dw,db) in zip(delta_weights,delta_biases):
				self.v_dw.append(np.full_like(dw,0))
				self.v_db.append(np.full_like(db,0))
				self.s_dw.append(np.full_like(dw,0))
				self.s_db.append(np.full_like(db,0))

		self.t+=1		
		
		for j, (delta_weight,delta_bias) in enumerate(zip(delta_weights,delta_biases)):
			self.v_dw[j]=self.B*self.v_dw[j]+(1-self.B)*(delta_weight)
			self.v_db[j]=self.B*self.v_db[j]+(1-self.B)*(delta_bias)	
			self.s_dw[j]=self.Y*self.s_dw[j]+(1-self.Y)*(delta_weight**2)
			self.s_db[j]=self.Y*self.s_db[j]+(1-self.Y)*(delta_bias**2)
			
		    #bias correction
			v_dw_correct=self.v_dw[j]/(1-self.B**self.t)
			v_db_correct=self.v_db[j]/(1-self.B**self.t)
			s_dw_correct=self.s_dw[j]/(1-self.Y**self.t)
			s_db_correct=self.s_db[j]/(1-self.Y**self.t)

			##update weights and biases
			weights[j]=weights[j]-self.learning_rate*(v_dw_correct/(np.sqrt(s_dw_correct+self.epsilon)))
			biases[j]=biases[j]-self.learning_rate*(v_db_correct/(np.sqrt(s_db_correct+self.epsilon)))
		    
		return weights,biases


		


def loss_mse(y, y_hat):
	'''
	Compute Mean Squared Error (MSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		MSE loss between y and y_hat.

	'''
	return np.mean((y-y_hat)**2)

	

def loss_regularization(weights, biases):
	'''
	Compute l2 regularization loss.

	Parameters
	----------
		weights and biases of the network.

	Returns
	----------
		l2 regularization loss 
	'''
	loss_reg=0
	for entry in weights:
		loss_reg=loss_reg+np.sum(np.power(entry,2))
	return loss_reg
	
def loss_fn(y, y_hat, weights, biases, lamda):
	'''s
	Compute loss =  loss_mse(..) + lamda * loss_regularization(..)

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1
		weights and biases of the network
		lamda: Regularization parameter

	Returns
	----------
		l2 regularization loss 
	'''
	loss=loss_mse(y,y_hat)
	reg=loss_regularization(weights,biases)
	loss_f=loss+ lamda*reg
	return loss_f
	

def rmse(y, y_hat):
	'''
	Compute Root Mean Squared Error (RMSE) loss betwee ground-truth and predicted values.

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		RMSE between y and y_hat.
	'''
	return np.sqrt(np.mean((y-y_hat)**2))


def cross_entropy_loss(y, y_hat):
	'''
	Compute cross entropy loss

	Parameters
	----------
		y : targets, numpy array of shape m x 1
		y_hat : predictions, numpy array of shape m x 1

	Returns
	----------
		cross entropy loss
	'''
	raise NotImplementedError

def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
	'''
	In this function, you will perform following steps:
		1. Run gradient descent algorithm for `max_epochs` epochs.
		2. For each bach of the training data
			1.1 Compute gradients
			1.2 Update weights and biases using step() of optimizer.
		3. Compute RMSE on dev data after running `max_epochs` epochs.

	Here we have added the code to loop over batches and perform backward pass
	for each batch in the loop.
	For this code also, you are free to heavily modify it.
	'''

	m = train_input.shape[0]
	epoch_losses = []
	for e in range(max_epochs):
		epoch_loss = 0.
		for i in range(0, m, batch_size):
			batch_input = train_input[i:i+batch_size]
			batch_target = train_target[i:i+batch_size]
			pred = net(batch_input)

			# Compute gradients of loss w.r.t. weights and biases
			dW, db = net.backward(batch_input, batch_target, lamda)

			# Get updated weights based on current weights and gradients
			weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

			# Update model's weights and biases
			net.weights = weights_updated
			net.biases = biases_updated

			# Compute loss for the batch
			batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			epoch_loss += batch_loss
		print(e,' ',epoch_loss)
		epoch_losses.append([e,epoch_loss])

			#print(e, i, rmse(batch_target, pred), batch_loss)
		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.
	epoch_losses = np.array(epoch_losses)
	p, q = epoch_losses.T
	plt.plot(p,q)
	plt.savefig('../train_64.png')

	dev_pred = net(dev_input)
	dev_rmse = rmse(dev_target, dev_pred)
	print('RMSE on dev data: {:.5f}'.format(dev_rmse))


def get_test_data_predictions(net, inputs):
	'''
	Perform forward pass on test data and get the final predictions that can
	be submitted on Kaggle.
	Write the final predictions to the part2.csv file.

	Parameters
	----------
		net : trained neural network
		inputs : test input, numpy array of shape m x d

	Returns
	----------
		predictions (optional): Predictions obtained from forward pass
								on test data, numpy array of shape m x 1
	'''
	return net(inputs)
	
	
	

def read_data():
	train = pd.read_csv('../regression/data/train.csv')	
	train = train.to_numpy()
	train_input = train[:,1:92]
	train_target = train[:,0:1]
	
	dev=pd.read_csv('../regression/data/dev.csv')
	dev=dev.to_numpy()
	dev_input=dev[:,1:92]
	dev_target=dev[:,0:1]
	testValues=pd.read_csv('../regression/data/test.csv')
	test_input=testValues.to_numpy()
    
	return train_input, train_target, dev_input, dev_target, test_input
	


def main():

	# Hyper-parameters 
	max_epochs = 100
	batch_size = 64
	learning_rate = 0.001
	num_layers = 3
	num_units = 16
	lamda = 0.1 # Regularization Parameter

	train_input, train_target, dev_input, dev_target, test_input = read_data()
	net = Net(num_layers, num_units) 
	optimizer = Optimizer(learning_rate)
	train(
		net, optimizer, lamda, batch_size, max_epochs,
		train_input, train_target,
		dev_input, dev_target
	)
	print(get_test_data_predictions(net, test_input))


if __name__ == '__main__':
	main()
