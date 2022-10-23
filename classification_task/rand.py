import numpy as np
import pandas as pd
t=np.array([[1000000,2,3],[9,6,8]])

z=np.array([[10000,50000,-98765],[6790,8897,8990]])

z=np.exp(z-np.max(z,axis=1,keepdims=True))
print(z)
        


