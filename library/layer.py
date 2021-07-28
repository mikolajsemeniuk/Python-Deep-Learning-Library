"""
input -> Linear -> Tanh -> Linear -> output
"""
from typing import Callable, Dict
import numpy as np
from tensor import Tensor

class Layer:
    def __init__(self) -> None:
        self.parameters: Dict[str, Tensor] = {}
        self.gradient: Dict[str, Tensor] = {}
    def forward(self, inputs: Tensor) -> Tensor:
        raise NotImplemented()  # type: ignore #
    def backward(self, inputs: Tensor) -> Tensor:
        raise NotImplemented()  # type: ignore #


class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        """
        inputs -> (batch_size, input_size)
        outputs -> (batch_size, output_size) 
        """
        super().__init__()
        self.parameters["weights"] = np.random.randn(input_size, output_size)
        self.parameters["bias"] = np.random.randn(output_size)
    def forward(self, inputs: Tensor) -> Tensor:
        """
        output = linear equation = inputs @ weights + bias
        """
        self.inputs = inputs 
        return inputs @ self.parameters["weights"] + self.parameters["bias"]
    def backward(self, gradient: Tensor) -> Tensor:
        """
        if y = f(x) and x = inputs @ weights + bias
        then dy/da = f'(x) * b
        and dy/db = f'(x) * a
        and dy/dc = f'(x)

        if y = f(x) and x = a @ b + c
        then dy/da = f'(x) @ b.T
        and dy/db =  a.T @ f'(x)
        and dy/dc = f'(x)
        """
        self.gradient["bias"] = np.sum(gradient, axis = 0)
        self.gradient["weights"] = self.inputs.T @ gradient
        return gradient @ self.parameters["weights"].T

class Activation(Layer):
    def __init__(self, 
        function: Callable[[Tensor], Tensor],
        function_prime: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self.function = function
        self.function_prime = function_prime
    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs
        return self.function(inputs)
    def backward(self, gradient: Tensor):
        """
        if y = f(x) and x = g(z)
        then dy/dz = f'(x) * g'(z)
        """
        return self.function_prime(self.inputs) * gradient


# @staticmethod
def tanh(tensor: Tensor) -> Tensor:
    return np.tanh(tensor)

def tanh_prime(tensor: Tensor) -> Tensor:
    y: Tensor = tanh(tensor)
    return 1 - y ** 2

class Tanh(Activation):
    def __init__(self):
        super().__init__(tanh, tanh_prime)

