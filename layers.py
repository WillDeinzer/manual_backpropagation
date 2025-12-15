import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        # Output (out_features, 1)
        # A = Wx, so (out_features, in_features) * (in_features, 1)
        self.weights = np.random.randn(out_features, in_features) * 0.01
        self.bias = np.random.randn(out_features) * 0.01
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, x):
        self.x = x
        return x @ self.weights.T + self.bias
    
    def backward(self, input_gradient):
        self.grad_weights = input_gradient.T @ self.x
        self.grad_bias = np.sum(input_gradient, axis=0)
        return input_gradient @ self.weights
    
class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask
    
    def backward(self, input_gradient):
        return input_gradient * self.mask