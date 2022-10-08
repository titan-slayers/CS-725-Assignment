from customDtype import custom
from baseClass import base
import random
class Neuron(base):
    def __init__(self,inp,activation='tanh',initType='uniform'):
        if initType == 'gauss':
            self.weights = [custom(random.gauss(0,0.5)) for i in range(inp)]
        else:
            self.weights = [custom(random.uniform(-1,1)) for i in range(inp)]
        self.bias = custom(0)
        self.activation = activation

    def __repr__(self) -> str:
        return f"{self.activation}Neuron({len(self.weights)})"

    def __call__(self, xis):
        assert type(xis)==list
        assert type(self.weights)==list
        out = sum(wi*xi for wi,xi in zip(self.weights,xis))+self.bias
        if self.activation == 'tanh':
            return out.tanh()
        elif self.activation == 'softmax':
            return out
        return out.relu()

    def parameters(self):
        return self.weights+[self.bias]
