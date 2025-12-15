class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for lin_layer in self.parameters:
            lin_layer.weights -= (self.lr * lin_layer.grad_weights)
            lin_layer.bias -= (self.lr * lin_layer.grad_bias)

class Adam:
    def __init__(self, lr):
        self.lr = lr