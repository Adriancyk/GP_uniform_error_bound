import gpytorch
import torch
import numpy as np

class GPModel(gpytorch.models.ExactGP): 
    def __init__(self, Xtr, Ytr, likelihood):
        super(GPModel, self).__init__(Xtr, Ytr, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=True)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class FitGP():
    def __init__(self, Xtr, Ytr, likelihood):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.likelihood = likelihood
        self.model = GPModel(Xtr, Ytr, likelihood)
        self.training_iter = 50

    def fit(self):
        self.model.train()
        self.likelihood.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        for i in range(self.training_iter):
            # Zero gradients from previous iteration
            self.optimizer.zero_grad()
            # Output from model
            output = self.model(self.Xtr)
            # Calc loss and backprop gradients
            loss = -self.mll(output, self.Ytr)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, self.training_iter, loss.item(),
                self.model.covar_module.base_kernel.lengthscale.item(),
                self.model.likelihood.noise.item()
            ))
            self.optimizer.step()


def gradestj(fun, x0, eps=1e-3):
    N, E = x0.shape

    xpme = np.tile(x0[:, :, np.newaxis], (1, 1, 2)) + np.eye(E)[:, :, np.newaxis] * np.ones((1, N, 2)) * eps
    xpme = xpme.transpose(2, 0, 1).reshape(E, E * N * 2)

    V = fun(xpme).reshape(E, E * N * 2).T
    dVdx = ((V[:E * N] - V[N * E:]) / (2 * eps)).reshape(N, E).T

    return dVdx