import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


input_nodes = 2
cell_weights = 2
output_nodes = 1
learning_rate = 0.5

# Setting the number of nodes in each input, hidden and output layers

# fgw = Forget gate's weights
fgw = np.random.randn(input_nodes, cell_weights).T

# igw = Input gate's weights
igw = np.random.randn(input_nodes, cell_weights).T

# ogw = Output gate's weights
ogw = np.random.randn(input_nodes, cell_weights).T

# cgw = Candidate gate's weights
cgw = np.random.randn(input_nodes, cell_weights).T

# l = weights from LSTM cells to output
l = np.random.randn(2, 1).T

# Default LSTM cell states are declared below
cellstate = [[1, 1] for j in range(100)]
cellstate = np.array(cellstate, dtype=float)
cellstate = np.array(cellstate, ndmin=2).T

# Defining the sigmoid activation function to be used at forget gates


def sigmoid_activation_function(x):
    sigmoid = 1/(1+np.exp(-x))
    return sigmoid

# Defining the tanh activation function to be used at input gates


def tanh_activation_function(x):
    tanh = 1 - np.square(np.tanh(x))
    return tanh


def ForgetGate(cellstate, input_from_gate, output=1):
    input_from_gate = np.dot(fgw, input_from_gate)
    input_from_gate = output * input_from_gate
    output_to_gate = sigmoid_activation_function(input_from_gate)
    cellstate = cellstate * output_to_gate


def InputGate(input_from_gate, output=1):
    input_from_gate_1 = (np.dot(igw, input_from_gate))*output
    input_from_gate_2 = np.dot(cgw, input_from_gate)*output
    output_to_gate = sigmoid_activation_function(
        input_from_gate_1) * tanh_activation_function(input_from_gate_2)
    cellstate = cellstate + output_to_gate


def OutputGate(input_from_gate, output=1):
    input_from_gate = np.dot(ogw, input_from_gate)
    input_from_gate = output * input_from_gate
    output_to_gate = sigmoid_activation_function(input_from_gate)
    output = tanh_activation_function(cellstate) * output_to_gate

    return output


def forward_propagation(inp_1, inp_2, inp_3):
    cellstate = [[1, 1] for j in range(len(inp_1[0]))]
    cellstate = np.array(cellstate, dtype=float)
    cellstate = np.array(cellstate, ndmin=2).T
    # Input is passed through first cell in lstm network
    ForgetGate(inp_1)
    InputGate(inp_1)
    output = OutputGate(inp_1)
    # Input is passed through second cell in lstm network
    ForgetGate(inp_2, output)
    InputGate(inp_2, output)
    output = OutputGate(inp_2, output)
    # Input is passed through third cell in lstm network
    ForgetGate(inp_3, output)
    InputGate(inp_3, output)
    output = OutputGate(inp_3, output)
    # Dot product of output weights and final cell's output
    f_input = np.dot(l, output)
    # output of the neural network
    f_output = sigmoid_activation_function(f_input)
    return f_output, output


def errors(targeted_output, f_output):
    error_in_output = targeted_output - f_output
    error_in_hiddenlayer = np.dot(l.T, error_in_output)
    return error_in_output, error_in_hiddenlayer


def Back_propagation(training_1, training_2, training_3, f_out, f_output, error_in_output,
                     error_in_cell):
    l = l + (learning_rate *
             np.dot((error_in_output * (1.0 - f_output)), f_out.T))
    fgw = fgw + \
        (learning_rate * np.dot((error_in_cell * f_out * (1.0 - f_out)), training_1.T))
    igw = igw + \
        (learning_rate * np.dot((error_in_cell * f_out * (1.0 - f_out)), training_2.T))
    cgw = cgw + \
        (learning_rate * np.dot((error_in_cell * f_out * (1.0 - f_out)), training_2.T))
    ogw = ogw + \
        (learning_rate * np.dot((error_in_cell * f_out * (1.0 - f_out)), training_3.T))


def training(training_1, training_2, training_3, targeted_variable):
    # convert lists to 2d arrays
    training_1 = np.array(training_1, ndmin=2).T
    training_2 = np.array(training_2, ndmin=2).T
    training_3 = np.array(training_3, ndmin=2).T
    targeted_variable = np.array(targeted_variable, ndmin=2).T

    # calling the forward propagation
    f_output, f_out = forward_propagation(
        training_1, training_2, training_3)

    # Calcualting the errors in output and cell
    error_in_output, error_in_cell = errors(targeted_variable, f_output)

    # Calling the back propagation after calculating the errors
    Back_propagation(training_1, training_2, training_3, f_out, f_output, error_in_output,
                          error_in_cell)

    return f_output


def testing(testing_1, testing_2, testing_3):
    # transpose input
    testing_1 = testing_1.T
    testing_2 = testing_2.T
    testing_3 = testing_3.T
    # Calling forward propagation for testing
    f_output, f_out = forward_propagation(testing_1, testing_2, testing_3)
    # return final input
    return f_output
