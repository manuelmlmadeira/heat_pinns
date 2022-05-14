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
# Problem Dimension (n)
input_size = 2 # (t,x1)
# Hyper-parameters for the PINN
alpha = 0.1 # see source for this problem

# DEFINE BOUNDARY/INITIAL CONDITIONS - this function is problem specific and is passed to identify if,
# in this problem, a given point is a boundary point (True) or collocation point (False).
def isBoundary(data_point):
    return data_point[0] == 0. or data_point[1] == 0. or data_point[1] == 1.


# PREPARE TEST DATASET
# Closed-form solution to heat equation in a bar
num_series_terms = 500  # number of Fourier series terms
num_points_test = 50  # number of points in each dimension
x = np.linspace(0, 1, num = num_points_test)  # x from 0 to 1 with num_points_test
t = np.linspace(0, 4, num = num_points_test)  # x from 0 to 4 with num_points_test

x_test = np.zeros((num_points_test ** 2, input_size))  # where the x for the test dataset are stored
y_test = np.zeros((num_points_test ** 2, 1))  # where the targets for the test dataset are stored
for i, x_i in enumerate(x):
    for j, t_j in enumerate(t):
        u_tjxi = 0
        # for each point in the grid (t,x), we compute a value for u(t,x) by a closed_form series (in this case,
        # we consider num_series_terms terms)
        for n in range(1,num_series_terms+1):
            u_tjxi += 200/(math.pi * n) * ((-1)**(n+1) + 1) * math.sin(n*math.pi*x_i) * math.exp(-alpha*(n**2)*(math.pi**2)*t_j)
        # store result
        x_test[i * num_points_test + j] = [t_j, x_i]
        y_test[i * num_points_test + j] = u_tjxi

# convert to torch variable
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)


# PREPARE TRAINING DATASET (following boundary conditions)
lower_limits = np.array([0, 0])  # lower limits for both variables (respectively)
upper_limits = np.array([4, 1])  # upper limits for both variables (respectively)
N_boundary = 1000  # Number of boundary points to be generated
N_collocation = 1000  # Number of collocation points to be generated

# generate boundary points
x_train_boundary = np.zeros((N_boundary, input_size))
y_train_boundary = np.zeros((N_boundary, 1))
for i in range(N_boundary):
    boundary = random.randint(0,2)  # We have 3 types of boundary points (two boundaries and 1 set of initial
    # points). We randomly pick one of those types and build a data point accordingly.
    if boundary == 0:  # initial points
        t = lower_limits[0]
        x = upper_limits[1] * random.random()
        u = 100
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
x_train_collocation = upper_limits * np.array(pyDOE.lhs(input_size, N_collocation)) # distribute data points
# by the domain using Latin Hypercube Sampling
y_train_collocation = np.zeros((N_collocation, 1))  # the target of collocation points is 0

# generate dataset and targets
x_train = torch.Tensor(np.concatenate((x_train_boundary,x_train_collocation), axis=0))
x_train.requires_grad = True # To train on them
y_train = torch.Tensor(np.concatenate((y_train_boundary,y_train_collocation), axis=0))



# Save datasets and problem data
PATH = "1D_simplebar_dataset"
torch.save({
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'input_size': input_size,
            'alpha': alpha,
            'N_boundary': N_boundary,
            'N_collocation': N_collocation,
            'N_test': num_points_test**2
            } , PATH)
