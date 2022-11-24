"""Regression using pytorch"""
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f

### Regression using nn
# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70],
                   [74, 66, 43],
                   [91, 87, 65],
                   [88, 134, 59],
                   [101, 44, 37],
                   [68, 96, 71],
                   [73, 66, 44],
                   [92, 87, 64],
                   [87, 135, 57],
                   [103, 43, 36],
                   [68, 97, 70]],
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119],
                    [57, 69],
                    [80, 102],
                    [118, 132],
                    [21, 38],
                    [104, 118],
                    [57, 69],
                    [82, 100],
                    [118, 134],
                    [20, 38],
                    [102, 120]],
                   dtype='float32')

# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

# Define dataset
train_ds = TensorDataset(inputs, targets)

# Define data loader
batch_size = 5
Train_dl = DataLoader(train_ds, batch_size, shuffle=True)
for Xb, Yb in Train_dl:
    # print('batch: batch_size = 5, Samples = 15, Total_batch = 3, each of 5 samples ')
    print("Features in a batch", Xb)
    print("Outputs in a batch", Yb)
    break                                       # Remove break to see all 3 batches

# Define model i.e. initializing weights and biases
Model = nn.Linear(3, 2)                         # list(Model.parameters()) to see weights and biases in a list

# Define loss function
Loss_fn = f.mse_loss

# Define optimizer Stochastic indicates that samples are selected in random batches instead of as a single group
Opt = torch.optim.SGD(Model.parameters(), lr=1e-5)

# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:
            # Generate predictions
            pred = model(xb)                     # Generate predictions i.e. y = x * w' + b

            # Calculate loss
            loss = loss_fn(pred, yb)

            # Compute gradients
            loss.backward()

            # Update parameters using gradients
            opt.step()

            # Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))

fit(100, Model, Loss_fn, Opt, Train_dl)

# Generate predictions
preds = Model(inputs)
print("Predictions", preds)
print("Targets", targets)



















# ### Regression from scratch
# # Input (temp, rainfall, humidity)
# inputs = np.array([[73, 67, 43],
#                    [91, 88, 64],
#                    [87, 134, 58],
#                    [102, 43, 37],
#                    [69, 96, 70]], dtype='float32')
#
# # Targets (apples, oranges)
# targets = np.array([[56, 70],
#                     [81, 101],
#                     [119, 133],
#                     [22, 37],
#                     [103, 119]], dtype='float32')
#
# # Convert inputs and targets to tensors
# inputs = torch.from_numpy(inputs)
# targets = torch.from_numpy(targets)
#
# # Weights and biases
# w = torch.randn(2, 3, requires_grad=True) # number of output x number of features
# b = torch.randn(2, requires_grad=True)
# print(w)
# print(b)
#
# def model(x):
#     return x @ w.t() + b # inputs matmul weights_transposed plus biases
#
# # Generate predictions
# preds = model(inputs)
# print(preds)
# print(targets)
#
# # MSE loss
# def mse(t1, t2):
#     diff = t1 - t2
#     return torch.sum(diff**2) / diff.numel()
#
# # # Compute loss
# # loss = mse(preds, targets)
# # print(loss)
# #
# # # Compute gradients
# # loss.backward()
# #
# # # Gradients for weights and biases
# # print(w.grad)
# # print(b.grad)
# #
# # # Adjust weight and biases
# # with torch.no_grad():
# #     w -= w.grad * 1e-5
# #     b -= b.grad * 1e-5
# #     w.grad.zero_()
# #     b.grad.zero_()
# #
# # print(w)
# # print(b)
# #
# # # Calculate loss
# # preds = model(inputs)
# # loss = mse(preds, targets)
# # print(loss)
#
# # Train for 100 epochs
# for i in range(100):
#     preds = model(inputs)
#     loss = mse(preds, targets)
#     loss.backward()
#     with torch.no_grad():
#         w -= w.grad * 1e-5
#         b -= b.grad * 1e-5
#         w.grad.zero_()
#         b.grad.zero_()
#
# # Calculate loss
# preds = model(inputs)
# loss = mse(preds, targets)
# print(loss)
#
# print(preds)
# print(targets)
