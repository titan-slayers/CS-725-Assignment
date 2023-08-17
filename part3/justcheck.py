import numpy as np
def derivative_relu(h):
		
		t=np.copy(h)
		print(t)
		t[t>=0]=1
		t[t<0]=0
		print(t)
		return t
h = np.array([[0.1,-0.9,-0.7,1],[0.8,-0.3,-0.6,0.8]])
t =(derivative_relu(h))  
m = np.multiply(h,t)  
print(m)

