"""

"""

from tensor import Tensor
from nn import NeuronalNetwork
from loss import Loss, TSE
from optimizer import Optimizer, SGD
from data import DataIterator, BatchIterator

def train(network: NeuronalNetwork,
        inputs: Tensor,
        targets: Tensor,
        epochs: int = 5000,
        iterator: DataIterator = BatchIterator(),
        loss: Loss = TSE(),
        optimizer: Optimizer = SGD()):
    for epoch in range(epochs):
        epoch_loss: float = 0.0
        for batch in iterator(inputs, targets):
            predicted = network.forward(batch.inputs)
            epoch_loss += loss.loss(predicted, batch.targets)
            gradient = loss.gradient(predicted, batch.targets)
            network.backward(gradient)
            optimizer.step(network)
        print(f'epoch: {epoch}, loss: {epoch_loss}')
