# Imports
import torch
import torch.nn as nn
from random_heat_datasets import isBoundary

# Create NN
class PINN(nn.Module):

    def __init__(self, h_sizes, out_size):
        super(PINN, self).__init__()
        # Hidden (first element in h_sizes list is the input size)
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        # Output
        self.out = nn.Linear(h_sizes[-1], out_size)

    def forward(self, input):
        # Feedforward
        for linear_layer in self.hidden:
            input = torch.tanh(linear_layer(input))  # why tanh? Used in PINN paper
        output = self.out(input)
        return output

    # Get the Jacobian of (the output of) the neural net wrt the inputs
    def compute_u_x(self, input):
        self.u_x = torch.autograd.functional.jacobian(self, input, create_graph=True)
        return self.u_x

    # Get the Hessian of (the output of) the neural net wrt the inputs
    def compute_u_xx(self, input):
        self.u_xx = torch.autograd.functional.hessian(self, input, create_graph=True)
        return self.u_xx

    # Compute the f(t,x) of the heat equation
    def f(self, input, alpha):
        u_input = self.compute_u_x(input)
        u_t = u_input[0][0] # The first coordinate from the Jacobian is wrt to t
        f = u_t # Note: in heat PDE, f(t,x1, ..., xn) = u_t - alpha*(u_x1x1 + ... + u_xnxn)
        for i in range(1, input.shape[0]):
            u_xx = self.compute_u_xx(input)[i, i] # The diagonal elements of the Hessian (except the 1st)
            # are wrt x1, ... xn
            f -= alpha * u_xx
        return f

    # To each data point, compute the right output
    # u(t_i, x_i) if boundary point
    # f(t_i, x_i) if collocation point
    def score_function(self, data, alpha):
        scores = torch.zeros((data.shape[0], 1))
        for idx, data_point in enumerate(data):
            if isBoundary(data_point):  # if data point is boundary point
                scores[idx] = self.forward(data_point)
            else: # if data points is collocation point
                scores[idx] = self.f(data_point, alpha)
        return scores
