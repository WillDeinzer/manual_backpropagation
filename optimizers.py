import numpy as np

class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for lin_layer in self.parameters:
            lin_layer.weights -= (self.lr * lin_layer.grad_weights)
            lin_layer.bias -= (self.lr * lin_layer.grad_bias)

# Note - Adam is very stable, and converges in a smaller number of epochs
# However, the penalty gets scaled differently per parameter, which leads to poor generalization
class Adam:
    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.layers = layers

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for layer in self.layers:
            for param_name in ["weights", "bias"]:
                parameter = getattr(layer, param_name)
                gradient = getattr(layer, "grad_" + param_name)

                key = id(parameter)

                if key not in self.m:
                    self.m[key] = np.zeros_like(parameter)
                    self.v[key] = np.zeros_like(parameter)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradient
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradient ** 2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                parameter -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

# Very few changes
class AdamW:
    def __init__(self, layers, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=1e-4):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.layers = layers
        # Add weight decay
        self.weight_decay = weight_decay

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1

        for layer in self.layers:
            for param_name in ["weights", "bias"]:
                parameter = getattr(layer, param_name)
                gradient = getattr(layer, "grad_" + param_name)

                key = id(parameter)

                if key not in self.m:
                    self.m[key] = np.zeros_like(parameter)
                    self.v[key] = np.zeros_like(parameter)

                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * gradient
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (gradient ** 2)

                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)

                parameter -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

                # Decay weights (not biases)
                if param_name == 'weights':
                    parameter -= (self.lr * self.weight_decay * parameter)