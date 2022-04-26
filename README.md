Python Version Used - 3.9.10

Required Packages:
    1. matplotlib
    2. numpy
    3. pandas
    4. scikit-learn

The training script is in lstm.py

The code can be run by opening Results.ipynb. In this notebook, the
train_on_dataset function is imported from lstm.py and called.

The train_on_dataset takes two inputs - The dataset URL and the name
of the feature column

The log file is stored in project_logs.txt

There is a shell script install_dependencies.txt which creates a python
virtual environment, activates it and installs the required packages as
listed above. After making the file executable with chmod on your system,
it can be run as follows:
$ . ./install_dependencies.txt