inputs = [1, 2, 3, 2.5]
weights = [[0.91, 0.61, 0.7, 1.55], [0.74, 2.31, 1.7, 0.34], [2.1, 2, 0.7, 0.52]]
bias = [2, 3, 0.5]

# print(list(temp))
layer_output = []
for neuron_weights, neuron_biases in zip(weights, bias):
    neuron_output = 0
    for n_input, n_weight in zip(inputs, neuron_weights):
        neuron_output += n_input * neuron_output
    neuron_output += neuron_biases
    layer_output.append(neuron_output)

print(layer_output)
