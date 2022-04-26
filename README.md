## Dataset Used:
1. https://personal.utdallas.edu/~axs210036/FB.csv
2. https://personal.utdallas.edu/~axs210036/SAGE.csv
3. https://personal.utdallas.edu/~axs210036/SPNS.csv

## Original Dataset Used:
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset?select=symbols_valid_meta.csv

## Python Version Used - 3.9.10

## Required Packages:
    1. matplotlib
    2. numpy
    3. pandas
    4. scikit-learn

## Files:
1. lstm.py - It contains all the functions to implement
the LSTM. The function (train_on_dataset) is used to collect the dataset 
and start the training finally. This function takes two inputs - The dataset URL and the name
of the feature column.

2. Results.ipynb - In this notebook, the train_on_dataset function is imported from 
lstm.py and called.

3. project_logs.txt - This contains the training and testing metrics.

4. install_dependencies.txt - This is a shell script that creates a python
virtual environment activates it, and installs the required packages as
listed above. 

## Steps to run:
1. Set up the environment using the install_dependencies.txt file. (execute the command below)
$ . ./install_dependencies.txt
2. Run the command below to look at the results for predictions on three stocks 
$ jupyter notebook Results.ipynb