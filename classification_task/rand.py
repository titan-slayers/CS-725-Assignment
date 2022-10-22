import numpy as np
import pandas as pd

train = pd.read_csv('../classification/data/train.csv')	
train = train.to_numpy()
train_input = train[:,1:92]
train_target = train[:,0:1]
y= np.shape(train_target)[0]
print(y)
train_tar=np.zeros((y,4))
print(train_tar[80,1])


for i in range(y):
    if train_target[i,0]=="Very Old":
        train_tar[i,0]=1
    if train_target[i,0]=="Old":
        train_tar[i,1]=1 
    if train_target[i,0]=="Recent":
        train_tar[i,2]=1  
    if train_target[i,0]=="Recent":
        train_tar[i,2]=1  
print(train_tar) 


t = np.array([[1,0,0,0],[0,1,0,0,]])
l=np.array([[0.8,0.3,0.6,0.6],[0.7,0.4,0.1,0.8]])
l=-np.log(l)
z=(l*t)
u=np.sum(z,axis=1,keepdims=True)
print(u)





