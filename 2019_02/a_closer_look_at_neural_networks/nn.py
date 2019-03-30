#!/usr/bin/env python3
"""Build simple neural networks using Python and Numpy.
"""
from pprint import pprint
from typing import Dict, List

import numpy as np


def sigmoid_activation(weighted_sum: float) -> float:
    """Sigmoid activation function.
    """
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


def initialize_neural_network(
    num_inputs: int,
    num_hidden_layers: int,
    num_nodes_per_hidden_layer: List[int],
    num_nodes_output: int,
) -> Dict:
    """Initializes a neural network given a set of parameters.
    """
    num_nodes_previous = num_inputs

    network = {}

    # Loop through each layer and randomly initialize the weights and biases
    # associated with each layer.
    for layer in range(num_hidden_layers + 1):
        if layer == num_hidden_layers:
            layer_name = "output"
            num_nodes = num_nodes_output
        else:
            layer_name = "layer_{}".format(layer + 1)
            num_nodes = num_nodes_per_hidden_layer[layer]

        # Initialize weights and bias for each node.
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = "node_{}".format(node + 1)
            network[layer_name][node_name] = {
                "weights": np.around(
                    np.random.uniform(size=num_nodes_previous), decimals=2
                ),
                "bias": np.around(np.random.uniform(size=1), decimals=2),
            }

        num_nodes_previous = num_nodes

    return network


def calculate_weighted_sum(
    inputs: np.ndarray, weights: np.ndarray, bias: np.ndarray
) -> np.ndarray:
    """Calculate weighted sum of inputs, given weights and bias.
    """
    return np.sum(inputs * weights) + bias


def main():
    """Build some neural networks from scratch.
    """
    x1 = 0.5
    w1 = 0.2
    x2 = 4.0
    w2 = 0.5
    b = 0.03

    z = x1 * w1 + x2 * w2 + b
    print("z: {:.2f}".format(z))

    a = sigmoid_activation(z)
    print("a: {:f}".format(a))

    print("sigmoid_activation(1000000)  = {:f}".format(sigmoid_activation(1000000)))
    print("sigmoid_activation(0.000001) = {:f}".format(sigmoid_activation(0.000001)))

    np.random.seed(2019)
    iris_network = initialize_neural_network(4, 1, [7], 3)
    network1 = initialize_neural_network(10, 5, [32, 32, 32, 32, 32], 2)
    mnist_network = initialize_neural_network(784, 2, [32, 32], 10)
    pprint(network1)

    input_values = np.around(np.random.uniform(size=10), decimals=2)

    print("\nInput values = {}".format(input_values))

    node_weights = network1["layer_1"]["node_1"]["weights"]
    node_bias = network1["layer_1"]["node_1"]["bias"]

    print("network1: Layer 1 - Node 1 - Weights: {}".format(node_weights))
    print("network1: Layer 1 - Node 1 - Bias:    {}".format(node_bias))

    weighted_sum_for_node = calculate_weighted_sum(
        input_values, node_weights, node_bias
    )
    print(
        "Weighted sum for Layer 1 - Node 1 = {}".format(
            np.around(weighted_sum_for_node[0], decimals=2)
        )
    )

    node_output_value = sigmoid_activation(weighted_sum_for_node)
    print(
        "Output value for Layer 1 - Node 1 = {}".format(
            np.around(node_output_value[0], decimals=2)
        )
    )


if __name__ == "__main__":
    main()
