"""
loss function measures how far are predicted results from actual
"""
import numpy as np
from tensor import Tensor

class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplemented() # type: ignore #
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        raise NotImplemented() # type: ignore #
"""
Total square error
"""
class TSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)
    def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predicted - actual)
