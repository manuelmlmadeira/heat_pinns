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
input_size = 3 #(t,x1, x2)
# Hyper-parameters for the PINN (see source for this problem in Results jupyter notebook)
alpha = 2
plate_length = 10
max_iter_time = 50
delta_x = 1
delta_t = (delta_x ** 2)/(4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)


# PREPARE TEST DATASET
# Solve Problem through the proposed method in source


## COPIED CODE FROM SOURCE
# Initialize solution: the grid of u(k, i, j), and x_test and y_test
u = np.empty((max_iter_time, plate_length, plate_length))
# Initial condition everywhere inside the grid
u_initial = 0
# Boundary conditions
u_top = 100.0
u_left = 0.0
u_bottom = 0.0
u_right = 0.0
# Set the initial condition
u.fill(u_initial)
# Set the boundary conditions
u[:, (plate_length-1):, :] = u_top
u[:, :, :1] = u_left
u[:, :1, 1:] = u_bottom
u[:, :, (plate_length-1):] = u_right

# (Compute the solution used the method in source)
def calculate(u):
    for k in range(0, max_iter_time-1, 1):
        for i in range(1, plate_length-1, delta_x):
            for j in range(1, plate_length-1, delta_x):
                u[k + 1, i, j] = gamma * (u[k][i+1][j] + u[k][i-1][j] + u[k][i][j+1] + u[k][i][j-1] - 4*u[k][i][j]) + u[k][i][j]
    return u
# Do the calculation here
u = calculate(u)

# Build test dataset (store the results above in x_text and y_test)
x = np.arange(0, plate_length)*delta_x
t = np.arange(0, max_iter_time)*delta_t
x_test = np.zeros((plate_length*plate_length*max_iter_time, input_size))
y_test = np.zeros((plate_length*plate_length*max_iter_time, 1))
for i, t_i in enumerate(t):
    for j, x1_j in enumerate(x):
        for k, x2_k in enumerate(x):
            x_test[i * plate_length**2 + j*plate_length + k] = [t_i, x1_j, x2_k]
            y_test[i * plate_length**2 + j*plate_length + k] = u[i,j,k]

# convert to torch variable
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

# DEFINE BOUNDARY/INITIAL CONDITIONS
def isBoundary(data_point):
    return data_point[0] == float(t_down_limit) or data_point[1] == float(x_down_limit) or data_point[1] == float(x_up_limit) or data_point[2] == float(x_down_limit) or data_point[2] == float(x_up_limit)



# PREPARE TRAINING DATASET (following boundary conditions)
t_down_limit = t[0]
t_up_limit = t[-1]
x_down_limit = x[0]
x_up_limit = x[-1]
lower_limits = np.array([t_down_limit, x_down_limit, x_down_limit])  # lower limits for variables (respectively)
upper_limits = np.array([t_up_limit, x_up_limit, x_up_limit]) # upper limits for both variables (respectively)
N_boundary = 1000  # Number of boundary points to be generated
N_collocation = 1000  # Number of collocation points to be generated

# generate boundary points
x_train_boundary = np.zeros((N_boundary,input_size))
y_train_boundary = np.zeros((N_boundary,1))
for i in range(N_boundary):
    boundary = random.randint(0,5) # We have 5 types of boundary points (4 boundaries and 1 set of initial
    # points). We randomly pick one of those types and build a data point accordingly.
    if boundary == 0: # boundary condition hot
        t = t_up_limit * random.random()
        x1 = x_up_limit
        x2 = x_up_limit * random.random()
        u = 100
    elif boundary == 1: # boundary condition cold
        t = t_up_limit * random.random()
        x1 = x_down_limit
        x2 = x_up_limit * random.random()
        u = 0
    elif boundary == 2: # boundary condition cold
        t = t_up_limit * random.random()
        x1 = x_up_limit * random.random()
        x2 = x_down_limit
        u = 0
    elif boundary == 3: # boundary condition cold
        t = t_up_limit * random.random()
        x1 = x_up_limit * random.random()
        x2 = x_up_limit
        u = 0
    else: # initial condition cold
        t = t_down_limit
        x1 = x_up_limit * random.random()
        x2 = x_up_limit * random.random()
        u = 0
    x_train_boundary[i, :] = [t, x1, x2]
    y_train_boundary[i] = u

# generate collocation points
x_train_collocation = upper_limits * np.array(pyDOE.lhs(input_size, N_collocation)) # distribute data points
# by the domain using Latin Hypercube Sampling
y_train_collocation = np.zeros((N_collocation, 1)) # the target of collocation points is 0

# generate dataset and targets
x_train = torch.Tensor(np.concatenate((x_train_boundary,x_train_collocation), axis=0))
x_train.requires_grad = True
y_train = torch.Tensor(np.concatenate((y_train_boundary,y_train_collocation), axis=0))


# Save datasets and problem data
PATH = "plate_dataset"
torch.save({
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'input_size': input_size,
            'alpha': alpha,
            'N_boundary': N_boundary,
            'N_collocation': N_collocation,
            'N_test': plate_length*plate_length*max_iter_time,
            't_down_limit': t_down_limit,
            't_up_limit': t_up_limit,
            'x_down_limit': x_down_limit,
            'x_up_limit': x_up_limit,
            } , PATH)
