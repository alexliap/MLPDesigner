# MLP Designer

A program where the user fills the parameters for the creation of a multi-layer perceptron and trains and validates it on either CPU or GPU.
Button colours and other stuff are not supported on MacOS, because the app was built on "tkinter" module and
not on "tkmacosx".

- The dataset that is going to be used for training must have the target values on the last column of the .csv file.
- The dataset of choice is split 80/20 to training and validation set respectively.
- Predict function not supported yet.
- The MLP model that is produced has a batch normalization layer before every linear layer by default.