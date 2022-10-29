import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
#from sklearn.utils import shuffle

#create shuffle function from sklearn
def shuffle(X, y, random_state=42):
    if random_state:
        np.random.seed(random_state)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]

# The seed will be fixed to 42 for this assigmnet.
np.random.seed(42)

NUM_FEATS = 90
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

    def transform(self,X):
        X =  X - self._mean
        return np.dot(X, self.features.T)

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)


mx = MinMaxScaler()
sc = StandardScaler()

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
        self.biases.append(np.random.uniform(-1, 1, size=(1, 1)))
        self.weights.append(np.random.uniform(-1, 1, size=(self.num_units[-1], 1)))
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
        activations = []
        activations.append(X)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            dropout_vector = np.random.binomial(1, 1 - DROPOUT, size=(1, b.shape[0]))
            h = np.dot(a, w) + b.T
            if i < len(self.weights)-1:
                a = self.activation(h)	
                #Single layer dropout
                if i==1:
                    a = a * dropout_vector/(1-DROPOUT)
                #print(np.sum(dropout_vector==0)/dropout_vector.shape[1])
                #print(np.sum(a==0)/a.shape[0])
                #print(a)
                activations.append(a)
            else: # No activation for the output layer
                a=h
        self.activations = activations
        self.a = a
        return a
				
    '''
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
        self.activations = activations
        self.a = a
        return [activations,a]
    '''

    def backward(self, X, y, lamda):
        '''
        Compute and return gradients loss with respect to weights and biases.
        (dL/dW and dL/db)

        Parameters
        ----------
            X : Input to the network, numpy array of shape m x d
            y : Output of the network, numpy array of shape m x 1
            lamda : Regularization parameter.

        Returnspatch_sklearn
        ----------
            del_W : derivative of loss w.r.t. all weight values (a list of matrices).
            del_b : derivative of loss w.r.t. all bias values (a list of vectors).

        Hint: You need to do a forward pass before performing backward pass.
        '''
        #answer,pred= self.forward(X) #forward pass
        answer = self.activations
        pred = self.a
        batch_size=y.shape[0]
        delY= 2./batch_size*(pred-y)
        #delY = -2.0*(y-pred)/batch_size
        del_W=[]
        del_B=[]
        for (w,b,a) in reversed(list(zip(self.weights,self.biases,answer))):
            #print(w.shape,a.shape,np.dot(a,w).shape,delY.shape)
            out = np.dot(a,w) + b.T
            if out.shape[1] != 1:
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

    epoch_losses = []
    for e in range(max_epochs):
        # Shuffle the training data
        train_input, train_target = shuffle(train_input, train_target)
        m = train_input.shape[0]
        epoch_loss = 0.
        print('Train shape',train_input.shape)
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
            #if(batch_loss>3):
                #print('Batch loss: {}'.format(batch_loss))
                #train_input = np.concatenate((train_input,batch_input),axis=0)
                #train_target = np.concatenate((train_target,batch_target),axis=0)
            epoch_loss += batch_loss

        print(e,' ',epoch_loss)
        epoch_losses.append([e,epoch_loss])
        dev_pred = net(dev_input)
        dev_pred = sc.inverse_transform(dev_pred)
        dev_target_inv = sc.inverse_transform(dev_target)
        #rounded_pred = np.vectorize(round)(dev_pred)
        print(dev_target-dev_pred)
        #print(dev_pred)
        dev_rmse = np.mean(np.abs(dev_target_inv-dev_pred))
        print('Dev MAE: %f' % dev_rmse)
        dev_rmse = rmse(dev_target_inv, dev_pred)
        print('Dev RMSE: %f' % dev_rmse)
        #df = np.concatenate((dev_input,dev_target_inv,dev_pred),axis=1)
        #df = pd.DataFrame(np.concatenate(np.array([i for i in range(1,dev_pred.shape[0]+1)]), dev_pred),axis=0)
        #df = pd.DataFrame(dev_pred,columns=['Predictions'])
        #df.index +=1
        #df.index.name = 'Id'
        #df.to_csv('dev_pred.csv',index=True,columns=['Predictions'])
        #if(dev_rmse<9.15):
        #    raise Exception('Dev RMSE: %f' % dev_rmse)
        #dev_rmse = rmse(dev_target, rounded_pred)
        #print('Rounded Dev RMSE: %f' % dev_rmse)

            #print(e, i, rmse(batch_target, pred), batch_loss)
        # Write any early stopping conditions required (only for Part 2)
        # Hint: You can also compute dev_rmse here and use it in the early
        # 		stopping condition.

    # After running `max_epochs` (for Part 1) epochs OR early stopping (for Part 2), compute the RMSE on dev data.


    dev_pred = net(dev_input)
    dev_pred = sc.inverse_transform(dev_pred)
    dev_target_inv = sc.inverse_transform(dev_target)
    dev_rmse = rmse(dev_target_inv, dev_pred)
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
    global NUM_FEATS

    train = pd.read_csv('../regression/data/train.csv')	
    #evenly sample train data based on first column	
    grouped_train = train.groupby(train['1'])
    ##get 1000 samples from each group
    grouped_train = grouped_train.apply(lambda x: x.sample(n=4, replace=True))
    grouped_train = grouped_train.reset_index(drop=True)
    #append to train data
    train = train.append(grouped_train)
    train = train.reset_index(drop=True)
    #train = grouped_train
    print(train.shape)
    train = train.to_numpy()
    train_input = train[:,1:92]
    train_input = mx.fit_transform(train_input)
    train_target = train[:,0:1] 
    train_target_unique = np.array(np.unique(train_target, return_counts=True)).T
    #print(train_target_unique)
    #save to csv
    df = pd.DataFrame(train_target_unique)
    #df.to_csv('train_target_unique.csv')
    train_target_absent = set([i for i in range(1922,2011)]) - set(np.unique(train_target)) 
    print(train_target_absent)
    print("Min and Max of train target",np.min(train_target),np.max(train_target))
    train_target = sc.fit_transform(train_target)
    #Subtract min from target
    #train_target = train_target - np.min(train_target)
	
    #dev=pd.read_csv('../regression/data/dev.csv')
    dev=pd.read_csv('../regression/data/dev.csv')
    dev=dev.to_numpy()
    dev_input=dev[:,1:92]
    dev_input = mx.transform(dev_input)
    dev_target=dev[:,0:1] 
    dev_target_unique = np.array(np.unique(dev_target, return_counts=True)).T
    #print(dev_target_unique)
    #save to csv
    df = pd.DataFrame(dev_target_unique)
    #df.to_csv('dev_target_unique.csv')
    print("Min and Max of dev target",np.min(dev_target),np.max(dev_target))
    dev_target = sc.transform(dev_target)
    #dev_target = dev_target - np.min(dev_target)
    testValues=pd.read_csv('../regression/data/test.csv')
    test_input=testValues.to_numpy()
    test_input = mx.transform(test_input)
    
    pca = PCA(n_components = 0.999)
    #pca.fit(train_input)
    
    train_input = pca.fit_transform(train_input)
    dev_input = pca.transform(dev_input)
    test_input = pca.transform(test_input)
    #print("Cumulative Variances (Percentage):")
    #print(np.cumsum(pca.explained_variance_ratio_ * 100))
    #print("Number of features:", pca.n_components_)
    #print("Number of samples:", pca.n_samples_)
    #print("Number of features during PCA Fit:", pca.n_features_in_)
    components = pca.n_components_
    print(f'Number of components: {components}')
	


    NUM_FEATS = components
	
	


    return train_input, train_target, dev_input, dev_target, test_input
	
def custom_round(i):
    if i<1921.5 or i>2010.5:
        if i<1922:
            return 1922
        else:
            return 2010
    return i


def main():

    # Hyper-parameters 
    global DROPOUT
    max_epochs = 8
    batch_size = 1
    learning_rate = 0.001
    num_units = [78,32,16]
    num_layers = len(num_units)
    DROPOUT = 0.0
    lamda = 0.0 # Regularization Parameter
    train_input, train_target, dev_input, dev_target, test_input = read_data()
    print(NUM_FEATS)
    net = Net(num_layers, num_units) 
    optimizer = Optimizer(learning_rate)
    train(
        net, optimizer, lamda, batch_size, max_epochs,
        train_input, train_target,
        dev_input, dev_target
    )
    #print(get_test_data_predictions(net, test_input))
    test_pred = get_test_data_predictions(net, test_input)
    test_pred = sc.inverse_transform(test_pred)
    df = pd.DataFrame(test_pred,columns=['Predictions'])
    df.index +=1
    df.index.name = 'Id'
    df.to_csv('22m0742.csv',index=True,columns=['Predictions'])



if __name__ == '__main__':
    main()
