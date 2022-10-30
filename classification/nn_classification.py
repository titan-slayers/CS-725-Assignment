import numpy as np
import pandas as pd

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)
NUM_FEATS = 90


def shuffle(X, y, random_state=42):
    if random_state:
        np.random.seed(random_state)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

DROPOUT = 0.0

class MinMaxScaler():
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.min(X, axis=0)
        self.max = np.max(X, axis=0)
        return self

    def transform(self, X):
        return (X - self.min) / (self.max - self.min)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * (self.max - self.min) + self.min
    
class StandardScaler():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X):
        return X * self.std + self.mean


class PCA():

    def __init__(self,n_components = None):
        self.percent_variance = n_components*100
        self.no_features = None
        self.features = None
        self.n_components_ = None
        self._mean = None

    def fit(self,X):
        self._mean = np.mean(X ,axis= 0)
        X_m = X - self._mean
        covarianceMatrix = np.cov(X_m ,rowvar = False)

        eigenVals , eigenVecs = np.linalg.eigh(covarianceMatrix)
        sorted_index = np.argsort(eigenVals)[::-1]
        

        eigenVals = eigenVals[sorted_index]
        eigenVecs = eigenVecs[sorted_index]

        total = sum(eigenVals)
        feature_variances = [(i / total)*100 for i in eigenVals]

        percent = 0
        upper_index = len(feature_variances)

        for i in range(0,len(feature_variances)):
            percent += feature_variances[i]
            if percent >= self.percent_variance:
                upper_index = i
                break

        self.no_features = upper_index + 1
        self.n_components_ = upper_index + 1
        self.features = eigenVecs[:upper_index + 1]
        columnnames = ['TimbreAvg'+str(i) for i in range(1,12+1)]
        columnnames.extend(['TimbreCovariance'+str(i) for i in range(1,78+1)])
        df = pd.DataFrame(self.features,columns = columnnames)
        df.index = ['PC'+str(i) for i in range(1,self.no_features+1)]
        df.to_csv('PCA_betas.csv')

    def transform(self,X):
        X =  X - self._mean
        return np.dot(X, self.features.T)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)


mx = MinMaxScaler()
sc = StandardScaler()


class Net(object):
    def __init__(self, num_layers, num_units):
        self.num_layers = num_layers
        self.num_units = num_units
        self.biases = []
        self.weights = []
        self.activations = []
        self.a = None
        for i in range(num_layers):
            if i==0:
				# Input layer
                self.weights.append(np.random.uniform(-1, 1, size=(NUM_FEATS, self.num_units[i])))
            else:
				# Hidden layer
                self.weights.append(np.random.uniform(-1, 1, size=(self.num_units[i-1], self.num_units[i])))
            self.biases.append(np.random.uniform(-1, 1, size=(self.num_units[i], 1)))

		# Output layer
        self.biases.append(np.random.uniform(-1, 1, size=(4, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units[-1], 4)))
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
        activations = []
        activations.append(a)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = np.dot(a, w) + b.T
            if i < len(self.weights)-1:
                a = self.activation(h)
                activations.append(a)
            else: 
                output_layer = self.softmax(h - np.max(h))
        self.activations = activations
        self.a = output_layer
        return output_layer


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
        answer = self.activations
        pred = self.a
        
       
        batch_size=y.shape[0]
        delY= pred-y
        
        
        del_W=[]
        del_B=[]
        for (w,b,a) in reversed(list(zip(self.weights,self.biases,answer))):
			#print(w.shape,a.shape,np.dot(a,w).shape,delY.shape)
            out = np.dot(a,w) + b.T
            if out.shape[1] != 4:
                delY = delY * (out > 0)
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
    #Cross entropy loss
    #return -np.sum(y * np.log(y_hat))
    y_hat=np.clip(y_hat,1e-7,1-1e-7)
    loss =  y*y_hat
    
    loss=(np.sum(loss,axis=1))
    loss = -np.log(loss)
    return np.mean(loss)

    
def train(
	net, optimizer, lamda, batch_size, max_epochs,
	train_input, train_target,
	dev_input, dev_target,
    test_input
):
    m = train_input.shape[0]

    for e in range(max_epochs):

		# Shuffle the training data
        train_input, train_target = shuffle(train_input, train_target)
        epoch_loss = 0.
        #correct = 0
        #incorrect = 0
        #Conf matrix for each epoch
        conf_matrix = np.zeros((4, 4))
        correct = 0
        incorrect = 0
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
        dev_pred = net(dev_input)
        #print(dev_pred)
        dev_ce = cross_entropy_loss(dev_target, dev_pred)
        #Fill in the confusion matrix
        for i in range(len(dev_pred)):
            conf_matrix[np.argmax(dev_target[i])][np.argmax(dev_pred[i])] += 1
            if(np.argmax(dev_target[i]) == np.argmax(dev_pred[i])):
                correct += 1
            else:
                incorrect += 1
        #Compute micro f1 score
        micro_f1 = 0
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for i in range(4):
            true_pos += conf_matrix[i][i]
            false_pos += np.sum(conf_matrix[:,i]) - conf_matrix[i][i]
            false_neg += np.sum(conf_matrix[i,:]) - conf_matrix[i][i]
        micro_f1 = true_pos / (true_pos + 0.5*(false_pos + false_neg))
        print('Micro F1 score on Dev = Accuracy = : ', micro_f1)
        print('Epoch: %d, Train Loss: %.4f, Dev Accuracy: %.4f, Confusion matrix: \n%s' % (e, dev_ce, correct/(correct+incorrect), conf_matrix))
        #Get predictions for test data
        test_pred = net(test_input)
        test_pred_labels = []
        for i in range(len(test_pred)):
            ind = np.argmax(test_pred[i])
            if(ind == 0):
                test_pred_labels.append('Very Old')
            elif(ind == 1):
                test_pred_labels.append('Old')
            elif(ind == 2):
                test_pred_labels.append('Recent')
            else:
                test_pred_labels.append('New')
        

        df = pd.DataFrame(test_pred_labels, columns=['Predictions'])
        df.index += 1
        df.index.name = 'Id'
        df.to_csv('22m0742.csv',index=True)
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
    global NUM_FEATS
    train = pd.read_csv('./classification/data/train.csv')	
    #print count of each class     
    #grouped_train = train.groupby(train['1'])
	##get 1000 samples from each group
    #grouped_train = grouped_train.apply(lambda x: x.sample(n=5000, replace=True))
    #grouped_train = grouped_train.reset_index(drop=True)
	#append to train data
    #train = train.append(grouped_train)
    #train = train.reset_index(drop=True)
    #print(train['1'].value_counts())
    train = train.to_numpy()
    train_input = train[:,1:92]
    train_input = mx.fit_transform(train_input)
    train_input = train_input.astype(np.float64)
    
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
    #print(train_tar)        
    dev = pd.read_csv('./classification/data/dev.csv')	
    dev = dev.to_numpy()
    dev_input = dev[:,1:92]
    dev_input = mx.transform(dev_input)
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
    testValues=pd.read_csv('./classification/data/test.csv')
    test_input=testValues.to_numpy()
    test_input = mx.transform(test_input)
    pca = PCA(n_components = 0.99)
    #pca.fit(train_input)
    train_input = pca.fit_transform(train_input)
    dev_input = pca.transform(dev_input)
    test_input = pca.transform(test_input)
    components = pca.n_components_
    print(f'Number of components: {components}')

    NUM_FEATS = components
    return train_input, train_tar, dev_input, dev_tar, test_input
    


def main():

    # Hyper-parameters 
    max_epochs = 19
    batch_size = 1
    learning_rate = 0.001
    num_units = [78]
    num_layers = len(num_units)
    lamda = 0.0 # Regularization Parameter

    train_input, train_target, dev_input, dev_target, test_input = read_data()
    net = Net(num_layers, num_units) 
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target,
        test_input
    )
    answer=[]
    z=(get_test_data_predictions(net, dev_input))
    print(np.argmax(z, axis=1))     



if __name__ == '__main__':
        main()