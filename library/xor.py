"""

"""
import numpy as np

from train import train
from nn import NeuronalNetwork
from layer import Linear, Tanh

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

network = NeuronalNetwork([
    Linear(input_size = 2, output_size = 2)
])

train(network, inputs, targets)

for x, y in zip(inputs, targets):
    predicted = network.forward(x)
    print(f'x: {x}, predicted: {predicted}, y: {y}')