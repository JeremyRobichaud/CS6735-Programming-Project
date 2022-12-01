# ANN with backpropagation
import sys
from math import exp
from random import seed
from random import random


def sigmoid_function(x):
    # exp(x) = e^x
    return 1 / (1 + exp(x))


class NeuralNetwork:

    class Node:
        def __init__(self, num_inputs, preset_weights=None):
            self._delta = None
            # +1 for k value
            self._weights = [random() for _ in range(num_inputs + 1)]
            if preset_weights and type(preset_weights) == list:
                self._weights = preset_weights

        def get_weights(self):
            return self._weights

        def get_weights_at(self, i):
            assert i < len(self._weights) - 1
            assert i >= 0
            return self._weights[i]

        def set_weights(self, new_weights):
            self._weights = new_weights

        def get_delta(self):
            return self._delta

        def set_delta(self, new_delta):
            self._delta = new_delta

        def output(self, inputs):
            assert len(inputs) == len(self._weights) - 1

            # Get b value
            retval = self._weights[-1]

            # w0a0 + w1a1 + ... + wNaN
            for i in range(len(self._weights) - 1):
                retval += inputs[i] * self._weights[i]

            # sig(w0a0 + w1a1 + ... + wNaN + b)
            return sigmoid_function(retval)

    def __init__(self, target_attribute, other_attributes, all_attribute_values, num_hidden_layers=1):

        self._t_attr = target_attribute
        self._o_attrs = other_attributes
        self._all_attr_values = all_attribute_values

        # Step 0 - Set Hidden Layer Amount and other variables
        self._num_input = len(other_attributes)
        self._num_h = int(self._num_input)
        self._num_output = len(all_attribute_values[target_attribute])

        # Step 1 - Create Neural Network

        # Each Hidden Layer will be connected to all inputs
        # So they will have num_inputs weights, for num_h nodes
        self._layers = [
            [
                self.Node(self._num_input) for _ in range(self._num_h)
            ] for _ in range(num_hidden_layers)
        ]

        # Similarly, each Output Layer will be connected to all Hidden Layers
        # So they will have num_h weights, for num_output nodes
        self._layers.append([self.Node(self._num_h) for _ in range(self._num_output)])

    def _feed_forward(self, row, layer_num=None, node_num=None):

        row = list(row)

        # We will transform the inputs into numbers
        inputs = [None for _ in row]
        for i in range(len(row)):
            inputs[i] = list(self._all_attr_values[self._o_attrs[i]]).index(row[i])

        # Travers through all layers with given inputs
        for i in range(len(self._layers)):
            cur_layers = self._layers[i]
            temp_input = inputs.copy()
            inputs = []
            for j in range(len(cur_layers)):
                cur_node = cur_layers[j]
                retval = cur_node.output(temp_input)
                inputs.append(retval)
                # print(i, layer_num, j, node_num)
                if i == layer_num and j == node_num:
                    return retval

        # Apply final input to output layer
        return inputs

    def _feed_backward(self, df, l_rate=0.1):
        testing_df_without_answers = df.copy().drop(columns=[self._t_attr])

        deviations = [[0 for _ in range(len(self._layers[i]))] for i in range(len(self._layers))]
        count = 0

        for index, row in testing_df_without_answers.iterrows():
            count += 1
            answer = df.loc[index][self._t_attr]
            answer_idx = list(self._all_attr_values[self._t_attr]).index(answer)
            answer_matrix = [0.0 for _ in self._all_attr_values[self._t_attr]]
            answer_matrix[answer_idx] = 1.0

            outputs = self._feed_forward(row)
            assert len(outputs) == len(answer_matrix)
            assert len(outputs) == len(self._layers[-1])

            # For each output unit k: Delta_K <- output_k(1-output_k)(true_k-output_k)
            for i in range(len(self._layers[-1])):
                node = self._layers[-1][i]
                node.set_delta(
                    outputs[i]*(1 - outputs[i])*(answer_matrix[i] - outputs[i])
                )

            # For each hidden unit h: Delta_h <- output_h(1-output_h)E(weight_hk * delta_k)
            for i in range(len(self._layers) - 1):
                layer_indx = (i * -1) - 2
                for j in range(len(self._layers[layer_indx])):
                    node = self._layers[layer_indx][j]
                    node_output = self._feed_forward(row, len(self._layers) + layer_indx, j)

                    summation = 0

                    # E(weight_hk * delta_k)
                    for prev_node in self._layers[layer_indx + 1]:
                        summation += prev_node.get_weights_at(j) * prev_node.get_delta()
                    node.set_delta(
                        node_output * (1 - node_output) * summation
                    )

            # For each weight, weight_i_j <- weight_i_j + muu * delta_j * input_i_j
            row = list(row)
            inputs = [None for _ in row]
            for i in range(len(row)):
                inputs[i] = list(self._all_attr_values[self._o_attrs[i]]).index(row[i])

            for i, current_layer in reversed(list(enumerate(self._layers))):
                for j in range(len(current_layer)):
                    node = current_layer[j]
                    current_weights = node.get_weights()
                    temp_weights = current_weights.copy()
                    for w in range(len(current_weights) - 1):
                        # muu * delta_j * input_i_j
                        muu = float(l_rate)
                        delta_j = float(node.get_delta())

                        if i > 1:
                            input_i_j = float(self._feed_forward(row, i-1, w))
                        else:
                            input_i_j = float(inputs[w])

                        deviations[i][j] += muu * delta_j * input_i_j

        for i, current_layer in reversed(list(enumerate(self._layers))):
            for j in range(len(current_layer)):
                node = current_layer[j]
                current_weights = node.get_weights()
                temp_weights = current_weights.copy()
                for w in range(len(current_weights) - 1):
                    # Take average of deviation
                    temp_weights[w] = current_weights[w] + (deviations[i][j] / count)

                temp_weights[-1] = current_weights[-1] + (deviations[i][j]/count)

                node.set_weights(temp_weights)

    def train(self, df, l_rate=0.1):

        self._feed_backward(df, l_rate)

        return self

    def classify(self, row):
        outputs = self._feed_forward(row)
        val, idx = max((val, idx) for (idx, val) in enumerate(outputs))
        return self._all_attr_values[self._t_attr][idx]


def compute(df, target_attribute, other_attributes, all_attribute_values):

    # Initialize Network
    network = NeuralNetwork(target_attribute, other_attributes, all_attribute_values, 5)

    epoch_num = 5
    # Mini-Batch
    batch_size = 100
    for _ in range(epoch_num):
        # Shuffle the df
        df = df.sample(frac=1)

        new_batch = 0
        counter = 1
        flag = True

        while flag:
            prev_batch = new_batch
            new_batch = counter * batch_size
            if new_batch > len(df.index):
                new_batch = len(df.index)
                flag = False
            network.train(df.iloc[prev_batch:new_batch], l_rate=0.1)
            counter += 1

    # Return Network
    return network

