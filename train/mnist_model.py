import numpy as np
from nn.abstract import NNetwork, GD
from nn.layers import Linear


class NNModel(NNetwork):
    def __init__(self):
        super(NNModel, self).__init__()
        self.layers = [
            Linear(784, 100),
            Linear(100, 100),
            Linear(100, 10)
        ]

    def forward(self, x: np.ndarray):
        out = None
        for layer in self.layers:
            out = self.Sigmoid(layer(x))
        return out


class SGD(GD):
    def optimize(self, model: NNModel, batch: np.ndarray):
        for x, y in batch:
            y_hat = model(x)
            model.backward(x, y, y_hat)
        for layer in model.layers:  # type: Linear
            layer.weights = layer.weights - self.lr * (layer.derived_weights / len(batch))
        model.reset_gradients()
