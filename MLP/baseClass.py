class base:
    def parameters(self):
        return []

    def reset_gradients(self):
        for neuron in self.params():
            neuron.gradient = 0.0

    