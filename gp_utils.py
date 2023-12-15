import gpytorch
import torch
import numpy as np

class GPModel(gpytorch.models.ExactGP): 
    def __init__(self, Xtr, Ytr, likelihood):
        super(GPModel, self).__init__(Xtr, Ytr, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=Xtr.shape[1]))

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
        self.training_iter = 1000

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
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, self.training_iter, loss.item(),
            #     self.model.covar_module.base_kernel.lengthscale.item(),
            #     self.model.likelihood.noise.item()
            # ))
            self.optimizer.step()


        # # Define the closure
        # def closure():
        #     self.optimizer.zero_grad()
        #     output = self.model(self.Xtr)
        #     loss = -self.mll(output, self.Ytr)
        #     return loss

        # # Run the optimizer
        # self.optimizer.step(closure)
                
        return self.model


def gradestj(fun, x0, eps=1e-3):
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
    N, E = x0.shape

    if N > E: # if N > E, transpose x0
        x0 = x0.T
    epsm = torch.tensor([[eps], [-eps]]).unsqueeze(2)
    xpme = torch.tile(x0[:, :], (2, 1, E)) + torch.kron(torch.eye(E), torch.ones((1, N))) * epsm
    # dim E x E x 2N
    xpmeT = torch.transpose(xpme, 0, 1)
    V = fun(xpmeT.reshape(E, E * N * 2).T) # dim 4N x 1
    dVdx = ((V[:E * N] - V[N * E:]) / (2 * eps)).reshape(E, N) # dim E x N
    return dVdx


def gradestj1(fun, x0, count=0, eps=1e-3):
    # decouple the task for original gradestj to avoid task misplaced
    # gradestj1 is for refGeneral function
    if count == 1:
        def ref_traj(tau):
            return 2*torch.sin(tau)
        return gradestj(ref_traj, x0)
    
    if x0.dim() == 1:
        x0 = x0.unsqueeze(0)
    N, E = x0.shape
    # if N > E: # if N > E, transpose x0
    #     x0 = x0.T
    epsm = torch.tensor([[eps], [-eps]]).unsqueeze(2)
    xpme = torch.tile(x0[:, :], (2, 1, E)) + torch.kron(torch.eye(E), torch.ones((1, N))) * epsm
    # dim E x E x 2N
    xpmeT = torch.transpose(xpme, 0, 1)
    V = fun(gradestj1, xpmeT.reshape(E, E * N * 2).T, count-1)
    dVdx = ((V[:,:E * N] - V[:,N * E:]) / (2 * eps)).reshape(E, N) # dim E x N
    return dVdx


def ndgridj(grid_min, grid_max, ns):
    # grid_min: 1 x E or scalar
    # grid_max: 1 x E or scalar
    # ns: 1 x E
    # Xtr: N x E

    D = ns.size
    if np.isscalar(grid_max):
        grid_max = np.tile(grid_max, (D))
    if np.isscalar(grid_min):
        grid_min = np.tile(grid_min, (D))

    x = torch.linspace(grid_min[0], grid_max[0], int(ns[0]))
    y = torch.linspace(grid_min[1], grid_max[1], int(ns[0]))
    Xtr = torch.meshgrid(x, y, indexing='ij')
    Xtr = torch.stack((Xtr[0].flatten(), Xtr[1].flatten()), axis=1)

    return Xtr



if __name__ == '__main__':
    def pdyn_f(x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)

        return 1 - torch.sin(x[:, 0]) + 1 / (1 + torch.exp(-x[:, 1]))
    E = 2
    Nte = 1e4
    XteMin = torch.tensor([-6.0, -4.0]) # x1_min, x2_min
    XteMax = torch.tensor([4.0, 4.0]) # x1_max, x2_max
    Ndte = int(Nte**(1/E)) 
    Nte = Ndte**E

    Xte = ndgridj(XteMin, XteMax, Ndte*torch.ones((E, 1)))

    test = gradestj(pdyn_f, Xte)
    np.savetxt('test.csv', test, delimiter=',')
    eps = 1e-3
    x = torch.tensor([eps, -eps])
    print(x.shape)
    print(torch.kron(torch.eye(2), torch.ones((1, 10000)))[0, :])

