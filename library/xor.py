"""

"""
import numpy as np

from train import train
from nn import NeuronalNetwork
from layer import Linear, Tanh # Tanh not needed right now


inputs = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])

targets = np.array([
    [1, 0],
    [0, 1],
    [0, 1],
    [1, 0]
])

# FIXME
print(f'\ninputs shape: {inputs.shape}, targets shape: {targets.shape}')

linear = Linear(input_size = 2, output_size = 2)

network = NeuronalNetwork([
    linear
])

train(network, inputs, targets)

# FIXME
print()
for x, y in zip(inputs, targets):
    predicted = network.forward(x)
    print(f'x: {x}, predicted: {predicted}, y: {y}')