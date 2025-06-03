import numpy as np
# np.random.seed(0)
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()
X = [[1, 2, 3, 2.5], [2.0, 1.0, -1.0, 2.0], [-1.5, 1.2, 2.7, 0.8]]

X,y = spiral_data(100, 3) 

class layer_Dense:
    def __init__(self, n_inputs, n_neurons):
#       multiplying with .10 gives us numbers closer to zero
        self.weights = .10 *np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_Relu:
    def forward(self, inputs):
       self.output = np.maximum(0, inputs) 
layer1 = layer_Dense(2, 5)
activation1 = Activation_Relu()
layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
