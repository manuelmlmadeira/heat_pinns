# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PINN import PINN
import random
from temppeak_datasets import isBoundary

# REPRODUCIBILITY
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
save_models = False # True if you want the initial and final models to be saved in this run

# Load datasets and problem data (diffusivity and dimension)
PATH = "./temppeak_dataset"
checkpoint = torch.load(PATH)
x_train = checkpoint['x_train']
y_train = checkpoint['y_train']
x_test = checkpoint['x_test']
y_test = checkpoint['y_test']
input_size = checkpoint['input_size']
alpha = checkpoint['alpha']
N_test = checkpoint['N_test']
N_boundary = checkpoint['N_boundary']
N_collocation = checkpoint['N_collocation']

# PINN
# Initialize hidden/latent solution, u(t,x), neural net
hidden_layers_size = [input_size, 20, 20]
output_size = 1 #u(t,x) outputs a scalar
u = PINN(h_sizes=hidden_layers_size, out_size=output_size)

# TRAINING
# Check accuracy of current model in training set
# I'm aware that when I do the last step (averaging MSE by the number of batches, l. 54) I'm not completely
# correct, since I'm giving the same "importance" to the last batch (smaller than
# than the others, unless I have a dataset size divisible by the chosen batch-size). Nevertheless,
# it is a good approximation, since total number of batches >>> 1.
# This holds both for check_train_MSE and check_test_MSE.
def check_train_MSE(data_loader, model):
    # Note: this function "compares" (MSE across all training data points)
    # u(t,x) with the boundary conditions in the boundary points and
    # f(t,x) with 0 in the collocation points
    model.eval()
    with torch.no_grad():
        MSE = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            scores = model.score_function(x, alpha)
            loss = criterion(scores, y)
            MSE += loss
    av_MSE = MSE/(batch_idx+1)
    model.train()
    return float(av_MSE)

# Check accuracy of current model in test set
def check_test_MSE(data_loader, model):
    # Note: this function "compares" (MSE across all data points)
    # u(t,x) with u(t,x) obtained from the closed_form in ALL testing points
    model.eval()
    with torch.no_grad():
        MSE = 0
        for batch_idx, (x, y) in enumerate(data_loader):
            scores = model(x)
            loss = criterion(scores, y)
            MSE += loss
    av_MSE = MSE / (batch_idx + 1)
    model.train()
    return float(av_MSE)


# Hyper-parameters
learning_rate = 0.01
num_epochs = 20
batch_size = 32

# Set datasets, loss and optimizer
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
criterion = nn.MSELoss()
optimizer = optim.Adam(u.parameters(), lr=learning_rate)

print("Start training!")
# Initial Model
print(f"Initial MSE - Train MSE: {check_train_MSE(train_loader,u)} - Test MSE: {check_test_MSE(test_loader, u)}")
if save_models:
    torch.save({
                "model_dict": u.state_dict(),
                "hidden_sizes": hidden_layers_size,
                "output_size": output_size
                }, "initial_model")

# Start training through the dataset
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # forward propagation
        scores = u.score_function(data, alpha)
        loss = criterion(scores, targets)

        # zero previous gradients
        optimizer.zero_grad()
        # back-propagation
        loss.backward()
        # gradient descent or adam step
        optimizer.step()

    print(f"Epoch: {epoch+1} - Train MSE: {check_train_MSE(train_loader,u)} - Test MSE: {check_test_MSE(test_loader, u)}")

if save_models:
    torch.save({
                "model_dict": u.state_dict(),
                "hidden_sizes": hidden_layers_size,
                "output_size": output_size
                }, "final_model1")