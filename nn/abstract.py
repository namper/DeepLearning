from abc import abstractmethod
from typing import Type

import numpy as np


class Activation:
    @abstractmethod
    def derivative(self, *args, **kwargs):
        pass


class Layer:
    def __init__(self, in_feature: int, out_feature: int, include_bias: bool):
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.includes_bias = include_bias

        self.activation = np.array([])
        self.z = np.array([])

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_gradient(self, *args, **kwargs):
        pass


class Sigmoid(Activation):
    @staticmethod
    def sigmoid(z: np.ndarray):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z: np.ndarray):
        _z = self.sigmoid(z)
        return _z * (1 - _z)

    def __call__(self, *args, **kwargs):
        derive: bool = kwargs.pop('derive')
        if derive:
            return self.derivative(*args, **kwargs)
        return self.sigmoid(*args, **kwargs)


class NNetwork:
    layers = []
    Sigmoid = Sigmoid()

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradient()

    def backward(self, x, y, out, *args, **kwargs):
        """ x: input y: label for x
            out: output

            Computing delta cost function
            𝛿_𝐿 =∇𝑎𝐶 ⊙ 𝜎′(𝑧_𝐿)
            Iterating over layers backwards
        """
        delta = (out - y) * self.Sigmoid(self.layers[-1].z, derive=True)
        backwards_layers = reversed(self.layers[1:])
        for order, layer in enumerate(backwards_layers):
            z_previous = self.layers[order + 1].z
            delta = layer.backward(delta, z_previous)
        last_layer = self.layers[0]
        last_layer.backward(delta, x)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if not hasattr(self, 'layers'):
            self.layers = []
        if isinstance(value, Layer):
            self.layers.append(value)


class GD:
    def __init__(self, lr):
        # learning rate
        self.lr = lr

    @abstractmethod
    def optimize(self, model: Type[NNetwork], batch: np.ndarray):
        pass
