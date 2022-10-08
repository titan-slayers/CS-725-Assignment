from operator import ne
from customDtype import custom
from baseClass import base
from neuron import Neuron
from layer import Layer
from softmaxLayer import softmaxLayer
class Network(base):
    def __init__(self,inp,op, *args, **kwargs):
        network_shape = [inp] + op
        self.layers = []
        for i in range(len(network_shape)-1):
            act = kwargs.get('activation','tanh')
            if i==len(network_shape)-2:
                act = "softmax"
            self.layers.append(Layer(network_shape[i],network_shape[i+1],activation=act))

    def __repr__(self) -> str:
        newline = '\n'
        return f"Parameters in Network = ({newline.join(str(len(layer.parameters())) for layer in self.layers)})"

    def __call__(self, xis):
        for layer in self.layers:
            xis = layer(xis)
        softmaxOut = softmaxLayer(xis)
        softmaxOut.softmax()
        return softmaxOut

    def parameters(self):
        out = []
        for layer in self.layers:
            out.extend(layer.parameters) 
        return out


    