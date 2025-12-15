from layers import Linear, ReLU
class MLPModel:
    def __init__(self):
        self.layers = [Linear(784, 128),
                       ReLU(),
                       Linear(128, 64),
                       ReLU(),
                       Linear(64, 10)]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, gradient):
        for i in range(len(self.layers) - 1, -1, -1):
            gradient = self.layers[i].backward(gradient)

    # Gets the linear layers for the optimizer
    @property
    def parameters(self):
        return [layer for layer in self.layers if isinstance(layer, Linear)]