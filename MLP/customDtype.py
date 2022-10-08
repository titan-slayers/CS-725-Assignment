from math import exp
from numpy import log
class custom:
    def __init__(self, val=0.0, prev=set()):
        self.val = val
        self.prev = prev
        self.gradient = 0.0
        self.bw = lambda: None

    def __repr__(self):
        return f"Value = {self.val} Gradient = {self.gradient}"

    def __add__(self, next):
        next = next if isinstance(next,custom) else custom(next)
        out = custom(self.val+next.val,(self,next))
        def bw():
            self.gradient += out.gradient
            next.gradient += out.gradient
        out.bw = bw
        return out

    def __radd__(self,next):
        return self+next

    def __sub__(self,next):
        next = next if isinstance(next,custom) else custom(next)
        out = custom(self.val-next.val,(self,next))
        def bw():
            self.gradient += out.gradient
            next.gradient -= out.gradient
        out.bw = bw
        return out

    def __rsub__(self,next):
        return next-self
    
    def __mul__(self,next):
        assert type(self)==custom
        next = next if isinstance(next,custom) else custom(next)
        out = custom(self.val*next.val,(self,next))
        def bw():
            self.gradient += next.val * out.gradient
            next.gradient += self.val * out.gradient
        out.bw = bw
        return out
        
    def __rmul__(self,next):
        return self*next

    def __neg__(self):
        return self*-1

    def __truediv__(self,next):
        return self*(next**-1)

    def __rtruediv__(self,next):
        return next*(self**-1)

    def __pow__(self,next):
        assert isinstance(next,(int,float)), "Power other than int/float"
        self = self if isinstance(self,custom) else custom(self)
        out = custom(self.val**next,(self,))
        def bw():
            self.gradient += next*(self.val**(next-1)) * out.gradient
        out.bw = bw
        return out

    def exp(self):
        self = self if isinstance(self, custom) else custom(self)
        out = custom(exp(self.val), (self,))
        def bw():
            self.gradient += out.val * out.gradient
        out.bw = bw
        return out

    def log(self):
        self = self if isinstance(self, custom) else custom(self)
        out = custom(log(self.val),(self,))
        def bw():
            self.gradient += (self.val)**-1 * out.gradient
        out.bw = bw
        return out

    def relu(self):
        self = self if isinstance(self,custom) else custom(self)
        out = custom(self.val if self.val >=0 else 0, (self,))
        def bw():
            self.gradient += (0 < out.val)*out.gradient
        out.bw = bw
        return out

    def tanh(self):
        self = self if isinstance(self,custom) else custom(self)
        out = custom((exp(2.0*self.val)-1)/(exp(2.0*self.val)+1), (self,))
        def bw():
            self.gradient += (1-(out.val**2.0))*out.gradient
        out.bw = bw
        return out

    def sigmoid(self):
        self = self if isinstance(self,custom) else custom(self)
        out = custom((1/(1+exp(-self.val))),(self,))
        def bw():
            self.gradient += out.val*(1-out.val)*out.gradient
        out.bw = bw
        return out

    def backprop(self):
        visited = set()
        topo = []
        def topological_sort(node):
            if node not in visited:
                visited.add(node)
                for neuron in node.prev:
                    topological_sort(neuron)
                topo.append(node)
        topological_sort(self)
        topo = reversed(topo)
        self.gradient = 1.0
        for neuron in topo:
            neuron.bw()
    
