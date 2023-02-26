import torch
from torch import nn
import matplotlib.pyplot as plt
from LinearRegressionModel import LRM

# DATA: PREPARING AND LOADING

# Machine learning is a game of 2 parts:
#  1. Get data into a numerical representation
#  2. Build a model to learn patterns in that numerical representation

# Create some known data using the linear regression formula. We'll use a linear regression formula to make a straight line with known parameters

# Create known parameters where the weight is b and the bias is a (Y = aX + b)

weight = 0.7
bias = 0.3

# Let's create some data

start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# print("X tensor: ", X[:10])
# print("y values: ", y[:10])
# print("Length of X: ", len(X))
# print("Length of y: ", len(y))

# Now we know our inputs, X, and we know our outputs, y. We need to make a neural network that will predict the value of the parameters based on this information.

# Let's create a training and testing set with our data.

train_split = int(0.8 * len(X))

X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_predictions(train_data=X_train,
                      train_labels=y_train,
                      test_data=X_test,
                      test_labels=y_test,
                      predictions=None):
    
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})

    plt.show();

# Build model

torch.manual_seed(42)

model_0 = LRM()

with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

# We need a loss function and an optimizer to start training
# A loss function calculates how far off your model is from the desired output and the optimizer modifies the model to minimize the loss

loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

# Setup training loop
#   0. Loop through data
#   1. Forward pass
#   2. Calculate the loss
#   3. Optimizer zero grad
#   4. Backward pass (backpropagation)
#   5. Optimizer step (gradient descent)

epochs = 1

# 0. Loop
for epoch in range(epochs):
    # Set the model to training mode
    model_0.train() # train mode in PyTorch sets all parameters that require gradients to require gradients

    #




