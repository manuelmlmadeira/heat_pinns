# Imports
import numpy as np
import torch
import torch.nn as nn
from simplebar_datasets import isBoundary
from scipy.interpolate import BarycentricInterpolator
import scipy.integrate as integrate

# Create NN
class PINN(nn.Module):

    def __init__(self, h_sizes, out_size, delta_t, alpha):
        super(PINN, self).__init__()
        # Initialization of relevant variables
        self.input_size = h_sizes[0]  # first element in hidden layers sizes is the input size)
        self.num_stages = out_size-1  # PINN has q+1 outputs
        self.delta_t = delta_t
        self.alpha = alpha
        A, b, self.c = self.butcher_tableau() # Compute implicit Runge-Kutta coefficients
        self.M = torch.tensor(np.concatenate((A, np.array([b]))))
        # Build network
        self.hidden = nn.ModuleList()
        for k in range(len(h_sizes) - 1):
            self.hidden.append(nn.Linear(h_sizes[k], h_sizes[k + 1]))
        # Output
        self.out = nn.Linear(h_sizes[-1], out_size)

    def get_times(self):
        return np.append(self.c*self.delta_t, self.delta_t)

    def butcher_tableau(self):
        roots, weights = np.polynomial.legendre.leggauss(self.num_stages)
        # The complete formula is C = 0.5 * (b - a) * roots + 0.5 * (b + a), with b = 1 and a = 0
        C = 0.5 * roots + 0.5  # coefficients c_j
        # The complete formula is C = 0.5 (b - a) * weights, with b = 1 and a = 0
        B = 0.5 * weights  # coefficients b_j

        interpolator_poly = BarycentricInterpolator(C) # the most stable polynomial interpolator I could find
        max_error = 0 # track if the algorithms remains stable
        A = np.zeros((self.num_stages, self.num_stages)) # coefficients a_{ij}
        # apply formula to obtain a_{ij} (see p. 43 in Iserles, Arieh (1996), "A First Course in the Numerical Analysis
        # of Differential Equations")
        for i in range(self.num_stages):
            y_to_interpolate = [float(j == i) for j in range(self.num_stages)] # 1 if i==j, 0 otherwise
            interpolator_poly.set_yi(y_to_interpolate) # only change y in interpolation, x remains
            for j in range(self.num_stages):
                a_ji, err = integrate.quad(interpolator_poly, 0, C[j])  # general purpose integrator, probably can be
                # optimized
                if err > max_error:
                    max_error = err
                A[j, i] = a_ji # fill matrix
        print(f"interpolation/integration for butcher tableau - max error: {max_error}")
        return A, B, C

    # Get the Hessian of (the output of) the neural net wrt the inputs
    # We use jacobian twice, because the hessian function only accepts scalar functions.
    def compute_u_x(self, x):
        self.u_x = torch.autograd.functional.jacobian(self, x, create_graph=True)
        #self.u_x = torch.squeeze(self.u_x)
        return self.u_x
    def compute_u_xx(self, x):
        self.u_xx = torch.autograd.functional.jacobian(self.compute_u_x, x, create_graph=True)
        #self.u_xx = torch.squeeze(self.u_xx)
        return self.u_xx

    def compute_N(self, input):
        N = torch.zeros((self.num_stages,1)) # each output
        hessians_list = self.compute_u_xx(input) # list of hessians from each u^{n+cj}
        # , i.e. each output (except the last we do not care about the last one, by definition) of PINN has one Hessian
        # wrt to inputs, which are stored in this variable.
        for i in range(self.num_stages):
            divergence_i = torch.sum(torch.diagonal(hessians_list[i]))  # pick Hessian i and sum all the elements from its diagonal
            N[i,:] = divergence_i  # store it in the output variable

        N *= -self.alpha
        return N


    # Add an "extra layer" via f
    def f(self, input):
        out_1 = torch.reshape(self(input), (self.num_stages+1, 1)) # convert output from NN into matrix
        N = self.compute_N(input)
        out_2 = out_1 + self.delta_t * torch.matmul(self.M.float(), N)
        return out_2.T

    def forward(self, input):
        # Feedforward
        for linear_layer in self.hidden:
            input = torch.tanh(linear_layer(input))  # why tanh? Used in PINN paper
        output = self.out(input)
        return output

    # To each data point, compute the right output
    # u(t_i, x_i) if boundary point
    # f(t_i, x_i) if collocation point
    def score_function(self, data):
        scores = torch.zeros((data.shape[0], self.num_stages+1))
        for idx, data_point in enumerate(data):
            if isBoundary(data_point):  # if data point is boundary point
                scores[idx,:] = self(data_point)
            else: # if data points is collocation point
                scores[idx,:] = self.f(data_point)
        return scores
