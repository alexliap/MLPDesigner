# MLP Designer

A program where the user fills the parameters for the creation of a multi-layer perceptron and trains and validates it on either CPU or GPU.

- The dataset that is going to be used for training must have the target values on the last column of the .csv file.
- The dataset of choice is split 60/40 to training and validation set respectively.
- Predict function not supported yet.
- The MLP model that is produced has a batch normaliztion layer before every linear layer by default.