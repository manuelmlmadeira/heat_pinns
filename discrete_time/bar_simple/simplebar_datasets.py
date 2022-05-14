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
# Problem Dimension (n-1) (excluding t, since it is fixed)
input_size = 1
# Hyper-parameters for the PINN
alpha = 0.1  # see source for this problem
x_boundaries = np.array([[0, 1]]) # each line of the matrix has the boundaries for each dimension
delta_t = 0.5  # interval in time
t_boundaries = np.array([0, delta_t])


# DEFINE BOUNDARY/INITIAL CONDITIONS - this function is problem specific and is passed to identify if,
# in this problem, a given point is a boundary point (True) or collocation point (False).
def isBoundary(data_point):
    return data_point[0] == x_boundaries[0, 0] or data_point[0] == x_boundaries[0,1]


# PREPARE TRAINING DATASET (following boundary conditions)
lower_limits = x_boundaries[:,0]  # upper limits of variables (respectively)
upper_limits = x_boundaries[:,1]  # upper limits of variables (respectively)
num_boundaries = input_size*2  # each dimension has two boundaries
N_collocation = 250  # Number of collocation points to be generated
N_boundary = num_boundaries  # one point per boundary

# generate collocation points
# distribute data points by the domain using Latin Hypercube Sampling
x_train_collocation = lower_limits + (upper_limits - lower_limits) * np.array(pyDOE.lhs(input_size, N_collocation))
y_train_collocation = 100 * np.ones((N_collocation, 1))  # targets start as the initial conditions
# generate boundary points
x_train_boundary = x_boundaries.T * np.ones((N_boundary, input_size))
y_train_boundary = np.zeros((N_boundary, 1))

# generate dataset and targets as torch tensors
x_train = torch.Tensor(np.concatenate((x_train_boundary,x_train_collocation), axis=0))
x_train.requires_grad = True # To train on them
y_train = torch.Tensor(np.concatenate((y_train_boundary,y_train_collocation), axis=0))


# PREPARE TEST DATASET
# Closed-form solution to heat equation in a bar
num_series_terms = 500  # number of Fourier series terms
num_points_test = 50  # number of points to test in x
x_axis = np.linspace(x_boundaries[0,0], x_boundaries[0,1], num = num_points_test)  # x from 0 to 1 with num_points_test
t_test = t_boundaries[0] + delta_t

x_test = np.zeros((np.size(x_axis), 1))  # where the t and x for the test dataset are stored
y_test = np.zeros((np.size(x_axis), 1)) # where the targets for the test dataset are stored
for i, x_i in enumerate(x_axis):
    u_tjxi = 0
    # for each point in the grid (t_test, x), we compute a value for u(t,x) by a closed_form series (in this case,
    # we consider num_series_terms terms)
    for n in range(1,num_series_terms+1):
        u_tjxi += 200/(math.pi * n) * ((-1)**(n+1) + 1) * math.sin(n*math.pi*x_i) * math.exp(-alpha*(n**2)*(math.pi**2)*t_test)
    # store result
    x_test[i] = [x_i]
    y_test[i] = u_tjxi

# convert to torch variable
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)


# Save datasets and problem data
PATH = "1D_simplebar_dataset"
torch.save({
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'input_size': input_size,
            'alpha': alpha,
            'x_boundaries': x_boundaries,
            't_boundaries': t_boundaries,
            'delta_t': delta_t,
            'N_boundary': N_boundary,
            'N_collocation': N_collocation,
            'N_test': np.size(x_axis)
            }, PATH)
