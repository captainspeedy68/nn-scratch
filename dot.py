import numpy as np
inputs = [[1, 2, 3, 2.5], [2.0,1.0,-1.0,2.0], [-1.5,1.2, 2.7,0.8]]
weights = [[0.2, 0.8, -.5, 1], [0.74, -.31, 0.7, 0.34], [1.1, 1.2, 0.7, 0.52]]
bias = [2, 3, 0.5]
# bias = 2.0
output =np.dot(inputs, np.array(weights).T) + bias 
print(output)
