from customDtype import custom
from baseClass import base
from neuron import Neuron
class softmaxLayer(base):
    def __init__(self,neurons):
        self.neurons = neurons
        def softmax():
            denom = 0.0
            out = []
            for neuron in self.neurons:
                neuron_exp = neuron.exp()
                out.append(neuron_exp)
                denom+=neuron_exp.val
            softmax_out = []
            print([x.val for x in out])
            print(denom)
            for neuron in out:
                softmax_one = neuron/denom
                softmax_out.append(softmax_one)
            self.neurons = softmax_out
            return
        self.softmax = softmax

    def __repr__(self) -> str:
        return f"Softmax Layer = ({', '.join(str(neuron) for neuron in self.neurons)})"

    def parameters(self):
        out = []
        for neuron in self.neurons:
            out.extend(neuron.parameters) 
        return out

    def backprop(self):
        for neuron in self.neurons:
            neuron.backprop()