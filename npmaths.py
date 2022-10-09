import numpy as np
import pandas as pd
train = pd.read_csv('regression/data/train.csv')
train=train.to_numpy()
train_input=train[:,1:92]
train_target=train[:,0:1]
print(train_input.shape)