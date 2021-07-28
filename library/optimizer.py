"""
We use optimizer to adjust the parameters based
on gradient obtained during backward propagation
"""

from nn import NeuronalNetwork

class Optimizer:
    def step(self, network: NeuronalNetwork) -> None:
        raise NotImplemented() # type: ignore #

class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
    def step(self, network: NeuronalNetwork) -> None:
        for parameter, gradient in network.parameters_and_gradients():
            parameter -= self.learning_rate * gradient

