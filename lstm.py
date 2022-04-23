import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# plotting of graphs
def plot_graphs(real_value, training_data, test_data):
    plt.plot(real_value, label="The actual value", color="red")
    plt.plot(training_data, label="Prediction on training data", color='blue')
    test_data = [j for j in test_data]
    # connecting training and testing lines
    test_data.insert(0, training_data[-1])
    # X values for testing prediction plot
    plt.plot([x for x in range(len(training_data) - 1, len(training_data) + len(test_data) - 1)], test_data, label="Prediction on testing data", color='green')
    plt.xlabel("Time period in Days")
    plt.ylabel("Stock Price")
    plt.title("Stock $ Prediction")
    plt.legend()
    plt.grid()
    plt.show()


def error_function(real_Value,predicted_Value):
    #squared error
    sse=((real_Value- predicted_Value)**2)
    #mean squared error
    mean_squared_error= np.mean(sse)
    #root mean squared error
    root_mean_squared_error=np.sqrt(mean_squared_error)
    
    return mean_squared_error,root_mean_squared_error

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


def InputGate(cellstate, input_from_gate, output=1):
    input_from_gate_1 = (np.dot(igw, input_from_gate))*output
    input_from_gate_2 = np.dot(cgw, input_from_gate)*output
    output_to_gate = sigmoid_activation_function(
        input_from_gate_1) * tanh_activation_function(input_from_gate_2)
    cellstate = cellstate + output_to_gate


def OutputGate(cellstate, input_from_gate, output=1):
    input_from_gate = np.dot(ogw, input_from_gate)
    input_from_gate = output * input_from_gate
    output_to_gate = sigmoid_activation_function(input_from_gate)
    output = tanh_activation_function(cellstate) * output_to_gate

    return output


def forward_propagation(cellstate, inp_1, inp_2, inp_3):
    cellstate = [[1, 1] for j in range(len(inp_1[0]))]
    cellstate = np.array(cellstate, dtype=float)
    cellstate = np.array(cellstate, ndmin=2).T
    # Input is passed through first cell in lstm network
    ForgetGate(cellstate, inp_1)
    InputGate(cellstate, inp_1)
    output = OutputGate(cellstate, inp_1)
    # Input is passed through second cell in lstm network
    ForgetGate(cellstate, inp_2, output)
    InputGate(cellstate, inp_2, output)
    output = OutputGate(cellstate, inp_2, output)
    # Input is passed through third cell in lstm network
    ForgetGate(cellstate, inp_3, output)
    InputGate(cellstate, inp_3, output)
    output = OutputGate(cellstate, inp_3, output)
    # Dot product of output weights and final cell's output
    f_input = np.dot(l, output)
    # output of the neural network
    f_output = sigmoid_activation_function(f_input)
    return f_output, output


def errors(targeted_output, f_output):
    error_in_output = targeted_output - f_output
    error_in_hiddenlayer = np.dot(l.T, error_in_output)
    return error_in_output, error_in_hiddenlayer


def Back_propagation(fgw, igw, cgw, ogw, l, training_1, training_2, training_3, f_out, f_output, error_in_output,
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
    
    return fgw, igw, cgw, ogw, l


def training(fgw, igw, cgw, ogw, l, cellstate, training_1, training_2, training_3, targeted_variable):
    # convert lists to 2d arrays
    training_1 = np.array(training_1, ndmin=2).T
    training_2 = np.array(training_2, ndmin=2).T
    training_3 = np.array(training_3, ndmin=2).T
    targeted_variable = np.array(targeted_variable, ndmin=2).T

    # calling the forward propagation
    f_output, f_out = forward_propagation(cellstate, 
        training_1, training_2, training_3)

    # Calcualting the errors in output and cell
    error_in_output, error_in_cell = errors(targeted_variable, f_output)

    # Calling the back propagation after calculating the errors
    fgw, igw, cgw, ogw, l = Back_propagation(fgw, igw, cgw, ogw, l, training_1, training_2, training_3, f_out, f_output, error_in_output,
                          error_in_cell)

    return f_output


def testing(cellstate, testing_1, testing_2, testing_3):
    # transpose input
    testing_1 = testing_1.T
    testing_2 = testing_2.T
    testing_3 = testing_3.T
    # Calling forward propagation for testing
    f_output, f_out = forward_propagation(cellstate, testing_1, testing_2, testing_3)
    # return final input
    return f_output

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


url="https://raw.githubusercontent.com/bmounikareddy98/Machine-learning-project/main/GOOG%20(4).csv"
data = pd.read_csv(url)
data = data['Adj Close']
#normalization value
n_value=1000
#1st two days
training_1 = [[data[j-6], data[j-5]] for j in range(len(data[:570])) if j >= 6]
#3rd and 4th day
training_2 = [[data[j-4], data[j-3]] for j in range(len(data[:570])) if j >= 6]
#5th and 6th day
training_3 = [[data[j-2], data[j-1]] for j in range(len(data[:570])) if j >= 6]
#7th day or targeted train_pred
training_y = [[j] for j in data[6:570]]

#convert into arrays
training_1 = np.array(training_1, dtype=float)
training_2 = np.array(training_2, dtype=float)
training_3 = np.array(training_3, dtype=float)
training_y = np.array(training_y, dtype=float)

# Normalize
training_1= training_1/n_value
training_2 = training_2/n_value
training_3 = training_3/n_value
training_y = training_y/n_value

# create neural networks

# number of training cycles
train_cycles = 100
# training the LSTM network
for c in range(train_cycles):
    print("Training cycle: "+str(c))
for n in training_1:
    training_prediction = training(fgw, igw, cgw, ogw, l, cellstate, training_1, training_2, training_3, training_y)

# Determinning errors

error_mse_train, error_rmse_train=error_function(training_y,training_prediction)
print("Mean squared error of train data: "+ str(error_mse_train))
print("Root mean squared error of train data: "+ str(error_rmse_train))
# de-Normalize
training_prediction = np.array(training_prediction, dtype=float)
training_prediction *=n_value
training_y *=n_value*10

# transpose
training_prediction = training_prediction.T




testing_1 = [[data[j - 6], data[j - 5]] for j in range(570, 670)]
testing_2 = [[data[j - 4], data[j - 3]] for j in range(570, 670)]
testing_3 = [[data[j - 2], data[j - 1]] for j in range(570, 670)]
testing_y = [[j] for j in data[570:670]]

testing_1 = np.array(testing_1, dtype=float)
testing_2 = np.array(testing_2, dtype=float)
testing_3 = np.array(testing_3, dtype=float)
testing_y = np.array(testing_y, dtype=float)


# Normalization

testing_1 = testing_1/n_value
testing_2 = testing_2/n_value
testing_3 = testing_3/n_value
testing_y = testing_y/n_value

# test_pred the network with unseen data
testing_prediction = testing(cellstate, testing_1, testing_2, testing_3)
testing_prediction = np.array(testing_prediction, dtype=float)

# print various accuracies
error_mse_test, error_rmse_test=error_function(testing_y, testing_prediction)
print("Mean squared error of test data: "+ str(error_mse_test))
print("Root mean squared error of test data: "+ str(error_rmse_test))

# de-Normalize data
testing_prediction = testing_prediction * n_value

testing_y = testing_y * n_value*10


# transplose test_pred results
testing_prediction = testing_prediction.T



# plotting training and test_pred results on same graph

#print(df.head())
data = data.to_frame()
#print(df['Adj Close'].values)

plot_graphs(data['Adj Close'].values[0:670], training_prediction, testing_prediction)