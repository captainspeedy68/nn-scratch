import numpy as np
# np.random.seed(0)
import nnfs
from nnfs.datasets import spiral, spiral_data
from numpy.random import sample
nnfs.init()


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
class Activation_SoftMax:
    def forward(self, inputs):
#       to keep the numbers between 0 to 1 and the sum will be 1
        exp_values = np.exp(inputs - np.max(inputs, axis= 1, keepdims=True))
        probablities = exp_values / np.sum(exp_values, axis = 1, keepdims=True)
        self.output = probablities

X, y = spiral_data(samples= 100, classes=3)
dense1 = layer_Dense(2,3) # 2 because there are only 2 inputs X and y
activation1 = Activation_Relu()
dense2 = layer_Dense(3, 3)
activation2 = Activation_SoftMax()


dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output[:5])
