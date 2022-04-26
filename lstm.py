from nbformat import write
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


###############################################################################
# function: write_logs function
#
# purpose : To log the training parameters and model accuracy
#           metrics
#
# inputs  : sURL - The dataset used
#           sColumn - The feature used
#           fTrainTestSplit - The train test split
#           iDataSize - The size of the dataset
#           fLearningRate - The learning rate
#           iEpochs - Number of iterations
#           fTrainMSE - The training MSE
#           fTrainRSq - The training R2 score
#           fTrainFVU - The training fraction of variance unexplained
#           fTestMSE - The test MSE
#           fRSqTest - The test R2 score
#           fFVUTest - The test fraction of variance unexplained
#
# outputs : all parameters written to a log file
###############################################################################
def write_logs(sURL, sColumn, fTrainTestSplit, iDataSize, fLearningRate,
               iEpochs, fTrainMSE, fTrainRSq, fTrainFVU, fTestMSE, fTestRSq,
               fTestFVU):
    # Opening log file
    with open("project_logs.txt", 'a+') as f:
        f.write("The dataset used: "+str(sURL.split('/')[-1])+"\n")
        f.write("The feature used: "+str(sColumn)+"\n")
        f.write("Train/Test Split: "+str(fTrainTestSplit)+"\n")
        f.write("Size of dataset: "+str(iDataSize)+"\n")
        f.write("Learning Rate Used: "+str(fLearningRate)+"\n")
        f.write("Number of iterations: "+str(iEpochs)+"\n")
        f.write("Training MSE: "+str(fTrainMSE)+"\n")
        f.write("Training R2 Score: "+str(fTrainRSq)+"\n")
        f.write("Training Fraction of Variance Unexplained: "+str(fTrainFVU)+"\n")
        f.write("Test MSE: "+str(fTestMSE)+"\n")
        f.write("\n\n")


###############################################################################
# function: plot_graphs
#
# purpose : To plot the prediction vs actual value graphs
#
# inputs  : oRealValNp - The target values
#           oTrainPredNp - The training predictions
#           oTestPredNp - The testing predictions
#
# outputs : plotted graphs
###############################################################################
def plot_graphs(oRealValNp, oTrainPredNp, oTestPredNp):
    plt.plot(oRealValNp, label="The actual value", color="red")
    plt.plot(oTrainPredNp, label="Prediction on training data", color='blue')
    oTestPredNp = [j for j in oTestPredNp]
    # connecting training and testing lines
    oTestPredNp.insert(0, oTrainPredNp[-1])
    # X values for testing prediction plot
    plt.plot([x for x in range(len(oTrainPredNp) - 1, len(oTrainPredNp) +
             len(oTestPredNp) - 1)], oTestPredNp, label="Prediction on testing data",
             color='green')
    plt.xlabel("Time period in Days")
    plt.ylabel("Stock Price")
    plt.title("Stock $ Prediction")
    plt.legend()
    plt.grid()
    plt.show()


###############################################################################
# function: error_function
#
# purpose : To calculate the MSE and RMSE
#
# inputs  : oTargetNp - The target values
#           oPredNp - The predicted values
#
# outputs : fMSE - The MSE value
#           fRMSE - The RMSE value
###############################################################################
def error_function(oTargetNp, oPredNp):
    # squared error
    oSSENp = ((oTargetNp - oPredNp)**2)
    # mean squared error
    fMSE = np.mean(oSSENp)
    # root mean squared error
    fRMSE = np.sqrt(fMSE)

    # Calculating the training R2 score and FVU
    fRSq = r2_score(oTargetNp, oPredNp.T)
    fFVU = 1 - fRSq

    return fMSE, fRMSE, fRSq, fFVU


###############################################################################
# function: sigmoid_activation_function
#
# purpose : To calculate sigmoid of input
#
# inputs  : oInpNp - The input value
#
# outputs : oSigOutNp - The value of sigmoid output
###############################################################################
def sigmoid_activation_function(oInpNp):
    oSigOutNp = 1/(1+np.exp(-oInpNp))
    return oSigOutNp


###############################################################################
# function: tanh_activation_function
#
# purpose : To calculate tanh of input
#
# inputs  : oInpNp - The input value
#
# outputs : oTanHOutNp - The value of tanh output
###############################################################################
def tanh_activation_function(oInpNp):
    oTanHOutNp = 1 - np.square(np.tanh(oInpNp))
    return oTanHOutNp


###############################################################################
# function: train_on_dataset
#
# purpose : To train the RNN on a given dataset
#
# inputs  : sURL - The path to the dataset
#           sColumn - The feature to be used for training
#
# outputs : The prediction vs target graphs
###############################################################################
def train_on_dataset(sURL, sColumn):

    ###############################################################################
    # function: forget_gate
    #
    # purpose : To create the first cell in LSTM network
    #
    # inputs  : oCellStateNp - The state of the cell
    #           oGateInput - The input from the gate
    #           oGateOutput - The output of the gate
    #
    # outputs : The updated cell state
    ###############################################################################
    def forget_gate(oCellStateNp, oGateInput, oGateOutput=1):
        oGateInput = np.dot(oFGWNp, oGateInput)
        oGateInput = oGateOutput * oGateInput
        oOutToGate = sigmoid_activation_function(oGateInput)
        oCellStateNp = oCellStateNp * oOutToGate

    ###############################################################################
    # function: input_gate
    #
    # purpose : To create the second cell in LSTM network
    #
    # inputs  : oCellStateNp - The state of the cell
    #           oGateInput - The input from the gate
    #           oGateOutput - The output of the gate
    #
    # outputs : The updated cell state
    ###############################################################################
    def input_gate(oCellStateNp, oGateInput, oGateOutput=1):
        oInpFromGate1 = (np.dot(oIGWNp, oGateInput))*oGateOutput
        oInpFromGate2 = np.dot(oCGWNp, oGateInput)*oGateOutput
        oOutToGate = sigmoid_activation_function(
            oInpFromGate1) * tanh_activation_function(oInpFromGate2)
        oCellStateNp = oCellStateNp + oOutToGate

    ###############################################################################
    # function: output_gate
    #
    # purpose : To create the third cell in LSTM network
    #
    # inputs  : oCellStateNp - The state of the cell
    #           oGateInput - The input from the gate
    #           oGateOutput - The output of the gate
    #
    # outputs : The updated cell state
    ###############################################################################
    def output_gate(oCellStateNp, oGateInput, oGateOutput=1):
        oGateInput = np.dot(oOGWNp, oGateInput)
        oGateInput = oGateOutput * oGateInput
        oOutToGate = sigmoid_activation_function(oGateInput)
        oGateOutput = tanh_activation_function(oCellStateNp) * oOutToGate

        return oGateOutput

    ###############################################################################
    # function: forward_propagation
    #
    # purpose : To perform the forward propagation step in the network
    #
    # inputs  : oCellStateNp - The state of the cell
    #           oTrainDay12 - Data of the first two days
    #           oTrainDay34 - Data of the third and fourth day
    #           oTrainDay56 - Data of the fifth and sixth day
    #
    # outputs : oFinalOut - Output of the neural network
    #           oCellOutput - Output of the final cell
    ###############################################################################
    def forward_propagation(oCellStateNp, oTrainDay12, oTrainDay34, oTrainDay56):
        oCellStateNp = [[1, 1] for j in range(len(oTrainDay12[0]))]
        oCellStateNp = np.array(oCellStateNp, dtype=float)
        oCellStateNp = np.array(oCellStateNp, ndmin=2).T
        # Input is passed through first cell in lstm network
        forget_gate(oCellStateNp, oTrainDay12)
        input_gate(oCellStateNp, oTrainDay12)
        oCellOutput = output_gate(oCellStateNp, oTrainDay12)
        # Input is passed through second cell in lstm network
        forget_gate(oCellStateNp, oTrainDay34, oCellOutput)
        input_gate(oCellStateNp, oTrainDay34, oCellOutput)
        oCellOutput = output_gate(oCellStateNp, oTrainDay34, oCellOutput)
        # Input is passed through third cell in lstm network
        forget_gate(oCellStateNp, oTrainDay56, oCellOutput)
        input_gate(oCellStateNp, oTrainDay56, oCellOutput)
        oCellOutput = output_gate(oCellStateNp, oTrainDay56, oCellOutput)
        # Dot product of output weights and final cell's output
        oFinalInp = np.dot(oLCWNp, oCellOutput)
        # output of the neural network
        oFinalOut = sigmoid_activation_function(oFinalInp)
        return oFinalOut, oCellOutput

    ###############################################################################
    # function: errors
    #
    # purpose : To calculate the errors in the network
    #
    # inputs  : oTargetNp - The value of the target variable
    #           oFinalOut - The output given by the network
    #
    # outputs : oOutputError - Error of the output layer
    #           oHiddenError - Error of the hidden layer
    ###############################################################################
    def errors(oTargetNp, oFinalOut):
        oOutputError = oTargetNp - oFinalOut
        oHiddenError = np.dot(oLCWNp.T, oOutputError)
        return oOutputError, oHiddenError

    ###############################################################################
    # function: back_propagation
    #
    # purpose : To perform the backward propagation step in the network
    #
    # inputs  : oFGWNp - Weights of the forget gate
    #           oIGWNp - Weights of the input gate
    #           oCGWNp - Weights of the candidate gate
    #           oOGWNp - Weights of the output gate
    #           oLCWNp - Weights from LSTM cell to output
    #           oTrainDay12 - Data of the first two days
    #           oTrainDay34 - Data of the third and fourth day
    #           oTrainDay56 - Data of the fifth and sixth day
    #           oCellOutput - Output of the final cell
    #           oFinalOut - Output of the neural network
    #           oOutputError - Error of the output layer
    #           oHiddenError - Error of the hidden layer
    #
    # outputs : oFGWNp - Updated weights of the forget gate
    #           oIGWNp - Updated weights of the input gate
    #           oCGWNp - Updated weights of the candidate gate
    #           oOGWNp - Updated weights of the output gate
    #           oLCWNp - Updated weights from LSTM cell to output
    ###############################################################################
    def back_propagation(oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp, oTrainDay12,
                         oTrainDay34, oTrainDay56, oCellOutput, oFinalOut, oOutputError,
                         oHiddenError):
        oLCWNp = oLCWNp + (fLearningRate *
                           np.dot((oOutputError * (1.0 - oFinalOut)), oCellOutput.T))
        oFGWNp = oFGWNp + \
            (fLearningRate * np.dot((oHiddenError * oCellOutput * (1.0 - oCellOutput)),
                                    oTrainDay12.T))
        oIGWNp = oIGWNp + \
            (fLearningRate * np.dot((oHiddenError * oCellOutput * (1.0 - oCellOutput)),
                                    oTrainDay34.T))
        oCGWNp = oCGWNp + \
            (fLearningRate * np.dot((oHiddenError * oCellOutput * (1.0 - oCellOutput)),
                                    oTrainDay34.T))
        oOGWNp = oOGWNp + \
            (fLearningRate * np.dot((oHiddenError * oCellOutput * (1.0 - oCellOutput)),
                                    oTrainDay56.T))

        return oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp

    ###############################################################################
    # function: training
    #
    # purpose : To train the network
    #
    # inputs  : oFGWNp - Weights of the forget gate
    #           oIGWNp - Weights of the input gate
    #           oCGWNp - Weights of the candidate gate
    #           oOGWNp - Weights of the output gate
    #           oLCWNp - Weights from LSTM cell to output
    #           oCellStateNp - The state of the cell
    #           oTrainDay12 - Data of the first two days
    #           oTrainDay34 - Data of the third and fourth day
    #           oTrainDay56 - Data of the fifth and sixth day
    #           oTrainY - Target variable data
    #
    # outputs : oFGWNp - Updated weights of the forget gate
    #           oIGWNp - Updated weights of the input gate
    #           oCGWNp - Updated weights of the candidate gate
    #           oOGWNp - Updated weights of the output gate
    #           oLCWNp - Updated weights from LSTM cell to output
    #           oFinalOut - Output of the neural network
    ###############################################################################
    def training(oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp, oCellStateNp, oTrainDay12,
                 oTrainDay34, oTrainDay56, oTrainY):
        # convert lists to 2d arrays
        oTrainDay12 = np.array(oTrainDay12, ndmin=2).T
        oTrainDay34 = np.array(oTrainDay34, ndmin=2).T
        oTrainDay56 = np.array(oTrainDay56, ndmin=2).T
        oTrainY = np.array(oTrainY, ndmin=2).T

        # calling the forward propagation
        oFinalOut, oCellOutput = forward_propagation(oCellStateNp,
                                                     oTrainDay12, oTrainDay34,
                                                     oTrainDay56)

        # Calculating the errors in output and cell
        oOutputError, oHiddenError = errors(oTrainY, oFinalOut)

        # Calling the back propagation after calculating the errors
        oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp = back_propagation(oFGWNp, oIGWNp,
                                                                  oCGWNp, oOGWNp,
                                                                  oLCWNp,
                                                                  oTrainDay12,
                                                                  oTrainDay34,
                                                                  oTrainDay56,
                                                                  oCellOutput,
                                                                  oFinalOut,
                                                                  oOutputError,
                                                                  oHiddenError)

        return oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp, oFinalOut

    ###############################################################################
    # function: testing
    #
    # purpose : To test the network
    #
    # inputs  : oCellStateNp - The state of the cell
    #           oTestDay12 - Test data of the first two days
    #           oTestDay34 - Test data of the third and fourth day
    #           oTestDay56 - Test data of the fifth and sixth day
    #
    # outputs : oFinalOut - Output of the neural network
    ###############################################################################
    def testing(oCellStateNp, oTestDay12, oTestDay34, oTestDay56):
        # transpose input
        oTestDay12 = oTestDay12.T
        oTestDay34 = oTestDay34.T
        oTestDay56 = oTestDay56.T
        # Calling forward propagation for testing
        oFinalOut, oCellOutput = forward_propagation(
            oCellStateNp, oTestDay12, oTestDay34, oTestDay56)
        # return final input
        return oFinalOut

    iInputNodes = 2
    iCellWeights = 2
    fLearningRate = 0.5
    fTrainTestSplit = 0.8
    iDataSize = 700

    # Number of train and test samples
    iTrainSamples = int(iDataSize * fTrainTestSplit)

    # Setting the number of nodes in each input, hidden and output layers

    # oFGWNp = Forget gate's weights
    oFGWNp = np.random.randn(iInputNodes, iCellWeights).T

    # oIGWNp = Input gate's weights
    oIGWNp = np.random.randn(iInputNodes, iCellWeights).T

    # oOGWNp = Output gate's weights
    oOGWNp = np.random.randn(iInputNodes, iCellWeights).T

    # oCGWNp = Candidate gate's weights
    oCGWNp = np.random.randn(iInputNodes, iCellWeights).T

    # oLCWNp = weights from LSTM cells to output
    oLCWNp = np.random.randn(2, 1).T

    # Default LSTM cell states are declared below
    oCellStateNp = [[1, 1] for j in range(100)]
    oCellStateNp = np.array(oCellStateNp, dtype=float)
    oCellStateNp = np.array(oCellStateNp, ndmin=2).T

    oDataDf = pd.read_csv(sURL)
    oDataDf = oDataDf[sColumn]
    # normalization value
    iNormVal = 1000
    # 1st two days
    oTrainDay12 = [[oDataDf[j-6], oDataDf[j-5]]
                   for j in range(len(oDataDf[:iTrainSamples])) if j >= 6]
    # 3rd and 4th day
    oTrainDay34 = [[oDataDf[j-4], oDataDf[j-3]]
                   for j in range(len(oDataDf[:iTrainSamples])) if j >= 6]
    # 5th and 6th day
    oTrainDay56 = [[oDataDf[j-2], oDataDf[j-1]]
                   for j in range(len(oDataDf[:iTrainSamples])) if j >= 6]
    # 7th day or targeted train_pred
    oTrainY = [[j] for j in oDataDf[6:iTrainSamples]]

    # convert into arrays
    oTrainDay12 = np.array(oTrainDay12, dtype=float)
    oTrainDay34 = np.array(oTrainDay34, dtype=float)
    oTrainDay56 = np.array(oTrainDay56, dtype=float)
    oTrainY = np.array(oTrainY, dtype=float)

    # Normalize
    oTrainDay12 = oTrainDay12/iNormVal
    oTrainDay34 = oTrainDay34/iNormVal
    oTrainDay56 = oTrainDay56/iNormVal
    oTrainY = oTrainY/iNormVal

    # create neural networks

    # number of training cycles
    iEpochs = 1000

    # training the LSTM network
    for iEpoch in range(iEpochs):
        print("Training cycle: ", iEpoch)
        oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp, oTrainPredNp = training(
            oFGWNp, oIGWNp, oCGWNp, oOGWNp, oLCWNp, oCellStateNp, oTrainDay12,
            oTrainDay34, oTrainDay56, oTrainY)

    # Determining errors

    fTrainMSE, fTrainRMSE, fTrainRSq, fTrainFVU = error_function(oTrainY,
                                                                 oTrainPredNp)
    print("Mean squared error of train data: " + str(fTrainMSE))
    print("Root mean squared error of train data: " + str(fTrainRMSE))
    print("R2 Score of train data: " + str(fTrainRSq))
    print("Fraction of Variance Unexplained of train data: " + str(fTrainFVU))
    # de-Normalize
    oTrainPredNp = np.array(oTrainPredNp, dtype=float)
    oTrainPredNp *= iNormVal
    oTrainY *= iNormVal*10

    # transpose
    oTrainPredNp = oTrainPredNp.T

    oTestDay12 = [[oDataDf[j - 6], oDataDf[j - 5]]
                  for j in range(iTrainSamples, iDataSize)]
    oTestDay34 = [[oDataDf[j - 4], oDataDf[j - 3]]
                  for j in range(iTrainSamples, iDataSize)]
    oTestDay56 = [[oDataDf[j - 2], oDataDf[j - 1]]
                  for j in range(iTrainSamples, iDataSize)]
    oTestY = [[j] for j in oDataDf[iTrainSamples:iDataSize]]

    oTestDay12 = np.array(oTestDay12, dtype=float)
    oTestDay34 = np.array(oTestDay34, dtype=float)
    oTestDay56 = np.array(oTestDay56, dtype=float)
    oTestY = np.array(oTestY, dtype=float)

    # Normalization

    oTestDay12 = oTestDay12/iNormVal
    oTestDay34 = oTestDay34/iNormVal
    oTestDay56 = oTestDay56/iNormVal
    oTestY = oTestY/iNormVal

    # test_pred the network with unseen data
    oTestPredNp = testing(oCellStateNp, oTestDay12, oTestDay34, oTestDay56)
    oTestPredNp = np.array(oTestPredNp, dtype=float)

    # print various accuracies
    fTestMSE, fTestRMSE, fTestRSq, fTestFVU = error_function(oTestY,
                                                             oTestPredNp)
    print("Mean squared error of test data: " + str(fTestMSE))
    print("Root mean squared error of test data: " + str(fTestRMSE))
    print("R2 Score of test data: " + str(fTestRSq))
    print("Fraction of Variance Unexplained of test data: " + str(fTestFVU))

    # de-Normalize data
    oTestPredNp = oTestPredNp * iNormVal

    oTestY = oTestY * iNormVal*10

    # transpose test_pred results
    oTestPredNp = oTestPredNp.T

    # Writing the logs to a log file
    write_logs(sURL, sColumn, fTrainTestSplit, iDataSize, fLearningRate,
               iEpochs, fTrainMSE, fTrainRSq, fTrainFVU, fTestMSE, fTestRSq,
               fTestFVU)

    # plotting training and test_pred results on same graph

    oDataDf = oDataDf.to_frame()

    plot_graphs(oDataDf[sColumn].values[0:670],
                oTrainPredNp, oTestPredNp)
