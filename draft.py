import torch
import gpytorch
import numpy as np
import math
import matplotlib.pyplot as plt
from gp_utils import *
from scipy.io import loadmat

# def pdyn_f(x):
#     if not isinstance(x, torch.Tensor):
#         x = torch.tensor(x)

#     return 1 - torch.sin(x[:, 0]) + 1 / (1 + torch.exp(-x[:, 1]))

# x0 = torch.ones((2, 10))
# x0[0, :] = torch.linspace(1, 10, 10)
# x0[1, :] = torch.linspace(11, 20, 10)
# print(x0)
# print(x0.shape)
# xpme = torch.tile(x0[:, :], (2, 1, 2))
# print(xpme.shape)
# print(xpme)

# x1 = torch.kron(torch.eye(2), torch.ones((1, 10)))

# print(x1.shape)
# print(x1)
# eps = 1e-3
# epsm = torch.tensor([[eps], [-eps]]).unsqueeze(2)
# print(epsm.shape)
# print(epsm)

# x2 = x1 * epsm
# print(x2.shape)
# print(x2)

# xpme = xpme + x2
# xpme = torch.transpose(xpme, 0, 1)
# print('xpme: ', xpme)
# print(xpme.reshape(2, 2 * 10 * 2))



# dVdx1 = ((V[:2 * 10] - V[2 * 10:]) / (2 * eps))
# print(dVdx1.shape)
# print(dVdx1)
# dVdx = ((V[:2 * 10] - V[2 * 10:]) / (2 * eps)).reshape(10, 2).T

# print(dVdx.shape)
# print(dVdx)

# x1 = torch.ones(4)
# x2 = torch.ones(4)
# print(x1*x2)

# Assume x is your tensor of shape (10000, 2) and y is your tensor of shape (2,)
# Assume x is your tensor of shape (10, 2) and y is your tensor of shape (1, 2)
# x = torch.ones(4, 2)
# y = torch.ones(1, 2)
# print(x)
# print(y)
# # Subtract y from x
# result = x - y

# print('Result:', result)
# deltaL = torch.tensor(0.01)
# print(torch.norm(2))
torch.manual_seed(0)
sn = 0.1
# train_x = torch.linspace(0, 1, 10)
# train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.1
mat = loadmat('training_data.mat')
train_x = torch.tensor(mat['train_x']).t().squeeze(-1)
train_y = torch.tensor(mat['train_y']).t().squeeze(-1)  # Transpose and unsqueeze
# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise = torch.tensor(sn)
model = ExactGPModel(train_x, train_y, likelihood)
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for i in range(500):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    optimizer.step()

print('ls: ', model.covar_module.base_kernel.lengthscale)
# print(model.likelihood.noise)
print("sf: ", model.covar_module.outputscale)