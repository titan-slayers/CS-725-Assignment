from customDtype import custom
from baseClass import base
from neuron import Neuron
class Layer(base):
    def __init__(self,inp,op, *args, **kwargs):
        self.neurons = [Neuron(inp,**kwargs) for _ in range(op)]

    def __repr__(self) -> str:
        return f"Network Layer = ({', '.join(str(neuron) for neuron in self.neurons)})"

    def __call__(self, xis):
        out = [neuron(xis) for neuron in self.neurons]
        if len(out) == 1:
            return out[0]
        return out

    def parameters(self):
        out = []
        for neuron in self.neurons:
            out.extend(neuron.parameters) 
        return out
