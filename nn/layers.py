import numpy as np

from nn.abstract import Layer, Sigmoid


class Linear(Layer):
    def __init__(self, in_feature: int, out_feature: int, include_bias: bool = True):
        super(Linear, self).__init__(in_feature, out_feature, include_bias)
        self.weights = np.random.randn(in_feature, out_feature)
        self.bias = np.random.randn(out_feature, 1)
        self.derived_weights = np.zeros(self.weights.shape)
        self.derived_bias = np.zeros(self.bias.shape)
        self.Sigmoid = Sigmoid()

    def forward(self, previous_activation: np.ndarray) -> np.ndarray:
        """Forwarding method compute's layer's term before activation"""
        self.activation = previous_activation
        self.z = self.weights @ previous_activation + self.bias
        return self.z

    def backward(self, delta_next: np.ndarray, previous_z: np.ndarray) -> np.ndarray:
        """Backward method to compute delta, here count start from last to first
            𝛿_𝑙 =( ( 𝑤_(𝑙+1) )⋅𝛿_(𝑙+1) ) ⊙ 𝜎′(𝑧_𝑙)
            where ⊙ is Hadamard product
        """
        # update weight && bias
        self.derived_weights += delta_next @ self.activation.T
        self.derived_bias += delta_next
        # return delta
        sigmoid_der = self.Sigmoid(previous_z, der=True)
        return (self.weights.T @ delta_next) * sigmoid_der

    def reset_gradient(self) -> None:
        """Set gradient to zero vector"""
        self.derived_weights = np.zeros(self.derived_weights.shape)
        self.derived_bias = np.zeros(self.derived_bias.shape)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __str__(self):
        return f"Linear Layer With Weights {self.in_feature} {self.out_feature}"
