# Imports
import numpy as np
import torch
import pyDOE
import random
import math

# REPRODUCIBILITY
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# PROBLEM DATA
# Problem Dimension
input_size = 2 #(t,x1)
# Hyper-parameters for the PINN
alpha = 1.11 * 10**(-4) # see source for this problem

# DEFINE BOUNDARY/INITIAL CONDITIONS - this function is problem specific and is passed to identify if,
# in this problem, a given point is a boundary point (True) or collocation point (False).
def isBoundary(data_point):
    return data_point[0] == 0. or data_point[1] == 0. or data_point[1] == 1.

# NO TEST SET IN THIS CASE

# PREPARE TRAINING DATASET (following boundary conditions)
lower_limits = np.array([0, 0])  # lower limits for both variables (respectively)
upper_limits = np.array([60, 1])  # upper limits for both variables (respectively)
N_boundary = 1000  # Number of boundary points to be generated
N_collocation = 1000  # Number of collocation points to be generated

# generate boundary points
x_train_boundary = np.zeros((N_boundary, input_size))
y_train_boundary = np.zeros((N_boundary, 1))
for i in range(N_boundary):
    boundary = random.randint(0, 2) # We have 3 types of boundary points (two boundaries and 1 set of initial
    # points). We randomly pick one of those types and build a data point accordingly.
    if boundary == 0:  # initial points
        t = lower_limits[0]
        x = upper_limits[1] * random.random()
        if 0. < x <= 0.1:  # different temperatures in different portions (see source)
            u = 88
        elif x <= 0.2:
            u = 33
        elif x <= 0.3:
            u = 70
        elif x <= 0.4:
            u = 11
        elif x <= 0.5:
            u = 3
        elif x <= 0.6:
            u = 75
        elif x <= 0.7:
            u = 55
        elif x <= 0.8:
            u = 45
        elif x <= 0.9:
            u = 90
        elif x < 1.:
            u = 60
        else:
            u = 0
    elif boundary == 1:  # boundary 1
        t = upper_limits[0] * random.random()
        x = lower_limits[1]
        u = 0
    else:  # boundary 2
        t = upper_limits[0] * random.random()
        x = upper_limits[1]
        u = 0
    # store points generated
    x_train_boundary[i, :] = [t, x]
    y_train_boundary[i] = u
# generate collocation points
x_train_collocation = upper_limits * np.array(pyDOE.lhs(input_size, N_collocation))  # distribute data points
# by the domain using Latin Hypercube Sampling
y_train_collocation = np.zeros((N_collocation, 1)) # the target of collocation points is 0

# generate dataset and targets
x_train = torch.Tensor(np.concatenate((x_train_boundary,x_train_collocation), axis=0))
x_train.requires_grad = True
y_train = torch.Tensor(np.concatenate((y_train_boundary,y_train_collocation), axis=0))



# Save datasets and problem data
PATH = "random_heat_dataset"
torch.save({
            'x_train': x_train,
            'y_train': y_train,
            #'x_test': x_test,
            #'y_test': y_test,
            'input_size': input_size,
            'alpha': alpha,
            'N_boundary': N_boundary,
            'N_collocation': N_collocation,
            #'N_test': num_points_test**2
            } , PATH)
