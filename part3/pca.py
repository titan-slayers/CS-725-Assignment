from cgi import test
import sys
import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def read_data():
    sc = StandardScaler()
    train = pd.read_csv('../regression/data/train.csv')	
    train = train.to_numpy()
    train_input = train[:,1:92]
    train_input = sc.fit_transform(train_input)
    train_target = train[:,0:1]
	
    dev=pd.read_csv('../regression/data/dev.csv')
    dev=dev.to_numpy()
    dev_input=dev[:,1:92]
    dev_input = sc.transform(dev_input)
    dev_target=dev[:,0:1]
    testValues=pd.read_csv('../regression/data/test.csv')
    test_input=testValues.to_numpy()
    test_input = sc.transform(test_input)

    pca = PCA(n_components = 0.90)
    pca.fit(train_input)
    print("Cumulative Variances (Percentage):")
    print(np.cumsum(pca.explained_variance_ratio_ * 100))
    components = len(pca.explained_variance_ratio_)
    print(f'Number of components: {components}')

    train_input = pca.fit_transform(train_input)
    dev_input = pca.transform(dev_input)
    test_input = pca.transform(test_input)

    return train_input, train_target, dev_input, dev_target, test_input

read_data()