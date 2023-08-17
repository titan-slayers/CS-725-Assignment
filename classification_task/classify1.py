from cgi import test
import sys
import os
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import StandardScaler

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
        h=h.astype(float)
        
        h=np.exp(h-np.max(h,axis=1,keepdims=True))
        
        h=h/np.sum(h,axis=1,keepdims=True)
        for en in h:
                if np.isnan(en).any():
                    print("nan value")
        
        
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
                output_layer = self.softmax(h - np.max(h))
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
               output=self.softmax(h)

        return [activations,output]
    def derivative_relu(self,h):
        t=np.copy(h)
            
        t[t>0]=1
        t[t<=0]=0
        return t    


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
        delY= pred-y
        
        
        del_W=[]
        del_B=[]
        for (w,a) in reversed(list(zip(self.weights,answer))):
          
            delW=np.dot(a.T,delY)+ (lamda*w)
           
            
            delB=np.sum(delY,axis=0)
            delB = np.reshape(delB,(len(delB),1))
            delX=np.dot(delY,w.T)
            delder = self.derivative_relu(a)
            delY=delX*delder
            #delY=delX
            del_W.append(delW)
            del_B.append(delB)
        del_W.reverse()
        del_B.reverse()
        return del_W,del_B


class Optimizer(object):
    def __init__(self, learning_rate,B=0.9,Y=0.999,epsilon=1e-8):

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
            #s_dw_correct=s_dw_correct.astype(float)
            s_db_correct=self.s_db[j]/(1-self.Y**self.t)
            #s_db_correct=s_db_correct.astype(float)
           
            

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
    
    y_hat=np.clip(y_hat,1e-7,1-1e-7)
    loss =  y*y_hat
    
    loss=(np.sum(loss,axis=1))
    loss = -np.log(loss)
    return np.mean(loss)
    
def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target
):
    m = train_input.shape[0]

    for e in range(max_epochs):
        epoch_loss = 0.
        #correct = 0
        #incorrect = 0
        for i in range(0, m, batch_size):
            batch_input = train_input[i:i+batch_size]
            batch_target = train_target[i:i+batch_size]
            pred = net(batch_input)

            # Compute gradients of loss w.r.t. weights and biases
            dW, db = net.backward(batch_input, batch_target, lamda)
            for en in dW:
                if np.isnan(en).any():
                    print('Epoch w',e)
            for ent in db:
                if np.isnan(ent).any():
                    print('Epoch w',e)        
            

            # Get updated weights based on current weights and gradients
            weights_updated, biases_updated = optimizer.step(net.weights, net.biases, dW, db)

            # Update model's weights and biases
            net.weights = weights_updated
            net.biases = biases_updated


            for entry in net.weights:
                if np.isnan(entry).any():
                    print('Epoch w',e)

            for entr in net.biases:
                if np.isnan(entr).any():
                    print('Epoch w',e)

    
        train_loss = cross_entropy_loss(batch_target,pred)
        print(train_loss)
        #print(dev_pred)
    dev_pred = net(dev_input)    
    dev_ce = cross_entropy_loss(dev_target, dev_pred)
    print("****************")
    print(dev_ce)

			# Compute loss for the batch
			#batch_loss = loss_fn(batch_target, pred, net.weights, net.biases, lamda)
			#correct += np.sum(pred==batch_target)
			#incorrect += np.sum(pred!=batch_target)
			#print(batch_target,pred)
			#epoch_loss += batch_loss
		#print(f'Accuracy = {correct/(correct+incorrect)}')
			#print(e, i, rmse(batch_target, batch_loss)
		# Write any early stopping conditions required (only for Part 2)
		# Hint: You can also compute dev_rmse here and use it in the early
		# 		stopping condition.

	# After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.


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
    sc = StandardScaler()
    train = pd.read_csv('../classification/data/train.csv')	
    train = train.to_numpy()
    train_input = train[:,1:92]
    train_input = sc.fit_transform(train_input)
    
    train_target = train[:,0:1]
    y= np.shape(train_target)[0]
    
    train_tar=np.zeros((y,4))
    

    for i in range(y):
        if train_target[i,0]=="Very Old":
            train_tar[i,0]=1
        if train_target[i,0]=="Old":
            train_tar[i,1]=1 
        if train_target[i,0]=="Recent":
            train_tar[i,2]=1  
        if train_target[i,0]=="New":
            train_tar[i,3]=1  
    print(train_tar)        
    dev = pd.read_csv('../classification/data/dev.csv')	
    dev = dev.to_numpy()
    dev_input = dev[:,1:92]
    dev_input = sc.transform(dev_input)
    dev_target = dev[:,0:1]
    x= np.shape(dev_target)[0]
    
    dev_tar=np.zeros((x,4))
    

    for i in range(x):
        if dev_target[i,0]=="Very Old":
            dev_tar[i,0]=1
        if dev_target[i,0]=="Old":
            dev_tar[i,1]=1 
        if dev_target[i,0]=="Recent":
            dev_tar[i,2]=1  
        if dev_target[i,0]=="New":
            dev_tar[i,3]=1  
    testValues=pd.read_csv('../classification/data/test.csv')
    test_input=testValues.to_numpy()
    
    return train_input, train_tar, dev_input, dev_tar, test_input
    


def main():

    # Hyper-parameters 
    max_epochs = 50
    batch_size = 256
    learning_rate = 0.001
    num_layers = 3
    num_units = 64
    lamda = 0.1 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units) 
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    answer=[]
    z=(get_test_data_predictions(net,train_input))
    t=(np.argmax(z, axis=1))  
    y=np.zeros((len(t),4))
    print(len(t))
    for i in range(len(t)):
        y[i,t[i]]=1
    count=0
    for i in range(len(y)):
        tu=0
        for j in range(0, 4):
            if (y[i][j] == train_target[i][j]):
                tu=tu+1
        if(tu==4):
            count=count+1
    print(count)  
    z=(get_test_data_predictions(net,dev_input))
    t=(np.argmax(z, axis=1))  
    y=np.zeros((len(t),4))
    print(len(t))
    for i in range(len(t)):
        y[i,t[i]]=1
    count=0
    for i in range(len(y)):
        tu=0
        for j in range(0, 4):
            if (y[i][j] == dev_target[i][j]):
                tu=tu+1
        if(tu==4):
            count=count+1
    print(count)  
    zp=(get_test_data_predictions(net,test_input))
    tp=(np.argmax(zp, axis=1))  
        
    df = pd.DataFrame(tp,columns=['Predictions'])
    df.index +=1
    df.index.name = 'Id'
    df = df.replace(to_replace=0,value="Very Old")
    df=df.replace(to_replace=1,value="Old")
    df=df.replace(to_replace=2,value="Recent")
    df=df.replace(to_replace=3,value="New")
    print(df)

    
    df.to_csv('test_pred.csv',index=True,columns=['Predictions'])   
   
                         

           




       



if __name__ == '__main__':
        main()
    
  
