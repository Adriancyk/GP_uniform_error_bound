import numpy as np
import math
import torch
import gpytorch
from gp_utils import *


Ntr = 100 # number of training points
Tsim = 30 # Simulation time
Nsim = 200 # Simulation steps
sn = 0.1 # Observation noise
E = 2 # State space dimension

x0 = np.transpose([0, 0]) # Initial state


pFeli_lam = np.ones((E-1, 1))
pFeLi_kc = 2


def pdyn_f(x):
    return 1 - torch.sin(x[:, 0]) + 1 / (1 + torch.exp(-x[:, 1]))
def pdyn_g(x):
    return 1


tau = 1e-8
delta = 0.01
deltaL = 0.01


# Generating Training Points
print('Generating Training Points...')

Ntr = int(np.floor(Ntr ** (1 / E)))**E
# Generate the grid of input features (Xtr)
grid_x = torch.linspace(0, 3, int(math.sqrt(Ntr)))
grid_y = torch.linspace(-3, 3, int(math.sqrt(Ntr)))
Xtr = torch.meshgrid(grid_x, grid_y)

Xtr = torch.stack((Xtr[0].flatten(), Xtr[1].flatten()), axis=1)
Ytr = pdyn_f(Xtr) + math.sqrt(sn) * torch.randn(Ntr)

print('Generating Testing Points...')
Nte = 1e4
XteMin = torch.tensor([-6.0, -4.0]) # x1_min, x2_min
XteMax = torch.tensor([4.0, 4.0]) # x1_max, x2_max
Ndte = int(Nte**(1/E)) 
Nte = Ndte**E

Xte = torch.meshgrid(
    torch.linspace(XteMin[0], XteMax[0], Ndte),
    torch.linspace(XteMin[1], XteMax[1], Ndte)
) 

Xte = torch.cat([x.reshape(-1, 1) for x in Xte], dim=1) # Xte = [x1, x2]
# np.savetxt('Xtr.csv', Xtr, delimiter=',')
Xte1 = Xte[:, 0]
Xte2 = Xte[:, 1]
Ntrajplot = 100

print('Initializing GP...')

likelihood = gpytorch.likelihoods.GaussianLikelihood()
likelihood.noise_covar.noise = torch.tensor(sn)
GP = FitGP(Xtr, Ytr, likelihood)

print('Fitting GP...')
model = GP.fit()

print('Setup Lyapunov Function Stability Test...')

Lf = max(math.sqrt(torch.norm(gradestj(pdyn_f, Xte)**2, dim=0)))
