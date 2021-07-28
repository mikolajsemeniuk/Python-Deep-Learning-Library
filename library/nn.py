"""
neuronal network is a collection of layers
"""

from typing import Iterator, Sequence, Tuple
from tensor import Tensor
from layer import Layer

class NeuronalNetwork:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers
    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    def backward(self, gradient: Tensor) -> Tensor:
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        return gradient

    def parameters_and_gradients(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for layer in self.layers:
            for name, parameter in layer.parameters.items():
                gradient = layer.gradient[name]
                yield parameter, gradient
