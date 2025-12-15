import numpy as np

class CrossEntropyLoss:
    def forward(self, logits, y_true):
        # Subtract the max to avoid numerical issues
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exps / np.sum(exps, axis=1, keepdims=True)
        self.y = y_true

        # Return the mean of CE loss
        return -np.sum(y_true * np.log(self.probs + 1e-15)) / self.y.shape[0]

    def backward(self):
        return (self.probs - self.y) / self.y.shape[0]