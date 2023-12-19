import numpy as np
import math
import torch
import gpytorch
from torchdiffeq import odeint
from scipy.spatial.distance import pdist
from scipy.spatial import ConvexHull
from scipy.io import loadmat
from scipy.integrate import solve_ivp
from shapely.geometry import Polygon

from gp_utils import *
from dyn import *
from ctrl import *
from refGeneral import *


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Ntr = 100 # number of training points
Tsim = 30 # Simulation time
Nsim = 200 # Simulation steps
sn = 0.1 # Observation noise
E = 2 # State space dimension

x0 = torch.tensor([0, 0]) # Initial state
x0 = x0.unsqueeze(0) # dim 1 x E
#ref = lambda t: refGeneral(t, E+1, lambda tau: 2*torch.sin(tau)) # Reference trajectory
def ref_traj(tau):
    return 2*torch.sin(tau)

ref = lambda t: refGeneral(t, E+1, ref_traj) # Reference trajectory



def pdyn_f(x):
    # x: N x E
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float64)

    return 1.0 - torch.sin(x[:, 0]).double() + 1.0 / (1.0 + torch.exp(-x[:, 1]).double())

def pdyn_g(x):
    return 1.0


tau = torch.tensor(1e-8)
delta = torch.tensor(0.01)
deltaL = torch.tensor(0.01)


# Generating Training Points
print('Generating Training Points...')

Ntr = int(np.floor(Ntr**(1/E)))**E
Xtr = ndgridj(torch.tensor([0, -3]), torch.tensor([3, 3]), math.sqrt(Ntr)*np.ones((E, 1)))
Ytr = pdyn_f(Xtr) + sn * torch.randn(Ntr)
mat = loadmat('trainingData.mat')
Xtr = torch.tensor(mat['Xtr']).t().squeeze(-1)
Ytr = torch.tensor(mat['Ytr']).t().squeeze(-1)
# save training points
np.savetxt('Xtr.csv', Xtr, delimiter=',')
np.savetxt('Ytr.csv', Ytr, delimiter=',')


# ==================== #
# Generating Testing Points
print('Generating Testing Points...')
Nte = 1e4
XteMin = torch.tensor([-6.0, -4.0]) # x1_min, x2_min
XteMax = torch.tensor([4.0, 4.0]) # x1_max, x2_max
Ndte = int(Nte**(1/E)) 
Nte = Ndte**E

Xte = ndgridj(XteMin, XteMax, Ndte*torch.ones((E, 1)))
Xte1 = Xte[:, 0]
Xte2 = Xte[:, 1]

# save testing points for checking
# np.savetxt('Xte.csv', Xte, delimiter=',')
# np.savetxt('Xte1.csv', Xte1, delimiter=',')
# np.savetxt('Xte2.csv', Xte2, delimiter=',')

Ntrajplot = 100

print('Initializing GP...')
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1e-6))
likelihood.noise_covar.noise = torch.tensor(sn)
GP = FitGP(Xtr, Ytr, likelihood)

print('Fitting GP...')
model = GP.fit()
ls = model.covar_module.base_kernel.lengthscale.squeeze() # lengthscale
sf = model.covar_module.outputscale # signal variance
kfcn = model.covar_module
print('ls: ', ls)
print('sf: ', sf)


# ==================== #
# Setup Lyapunov Function Stability Test
print('Setup Lyapunov Function Stability Test...')
doing_test = False
Lf = torch.max(torch.sqrt(torch.norm(gradestj(pdyn_f, Xte)**2, dim=0)))
Lk = torch.norm(sf**2*np.exp(-0.5)/ls)

print('Lf: ', Lf)
print('Lk: ', Lk)

# ==================== #
def k(x, xp):
    return (sf**2) * torch.exp(-0.5 * torch.sum((x - xp)**2 / (ls**2), dim=1))

def dkdxi(x, xp, i):
    if len(xp.shape) == 1:
        xp.unsqueeze_(0)
    return -(x[:, i] - xp[:, i]) / (ls[i]**2) * k(x, xp)

def ddkdxidxpi(x, xp, i):
    if len(xp.shape) == 1:
        xp.unsqueeze_(0)
    return (ls[i]**(-2)) * k(x, xp) + (x[:, i] - xp[:, i]) / (ls[i]**2) * dkdxi(x, xp, i)

def dddkdxidxpi(x, xp, i):
    if len(xp.shape) == 1:
        xp.unsqueeze_(0)
    return -(ls[i]**(-2)) * dkdxi(x, xp, i) - (ls[i]**(-2)) * dkdxi(x, xp, i) + (x[:, i] - xp[:, i]) / (ls[i]**2) * ddkdxidxpi(x, xp, i)

r = max(pdist(Xte))
Lfs = torch.zeros((E, 1))
if doing_test:
    for e in range(E):
        maxk = max(ddkdxidxpi(Xte, Xte, e))
        Lkds = torch.zeros(Nte, 1)
        for nte in range(Nte):
            Lkds[nte] = max(dddkdxidxpi(Xte, Xte[nte, :], e))
        Lkd = max(Lkds)
        Lfs[e] = torch.sqrt(2*torch.log(2*E/deltaL))*maxk + 12*torch.sqrt(torch.tensor(6*E))*max(maxk, torch.sqrt(r*Lkd))
    # temp = kfcn(Xtr).evaluate()+sn**2*torch.eye(Ntr)
    Lfh = torch.norm(Lfs)
    Lnu = Lk*torch.sqrt(torch.tensor(Ntr))*torch.norm((kfcn(Xtr).evaluate()+sn**2*torch.eye(Ntr))*Ytr)
    omega = torch.sqrt(2*tau*Lk*(1 + Ntr*torch.norm(kfcn(Xtr).evaluate()+sn**2*torch.eye(Ntr))*sf**2))
    beta = 2*torch.log((1 + ((max(XteMax) - min(XteMin))/tau))**E/delta)
    gamma = tau*(Lnu + Lfh) +torch.sqrt(beta)*omega
    # Save parameters to txt file
    with open('parameters.txt', 'w') as f:
        f.write(f'Lfh: {Lfh}\n')
        f.write(f'Lnu: {Lnu}\n')
        f.write(f'omega: {omega}\n')
        f.write(f'beta: {beta}\n')
        f.write(f'gamma: {gamma}\n')
else:
    import ast
    # Initialize a dictionary to store the parameters
    parameters = {}
    # Open the text file and read the parameters
    with open('parameters.txt', 'r') as f:
        for line in f:
            # Split the line into name and value
            name, value = line.strip().split(': ')
            # Convert the value to a float and store it in the dictionary
            value = ast.literal_eval(value.replace('tensor', ''))
            parameters[name] = torch.tensor(value)

    # Now you can access the parameters like this:
    Lfh = parameters['Lfh']
    Lnu = parameters['Lnu']
    omega = parameters['omega']
    beta = parameters['beta']
    gamma = parameters['gamma']
    print("Precomputed parameters have been successfully loaded.")

class pFeli:
    def __init__(self, pdyn_f, pdyn_g):
        self.kc = 2.
        self.lam = 1.
        self.f = pdyn_f
        self.g = pdyn_g

pFeLi = pFeli(pdyn_f, pdyn_g)

model.eval()
likelihood.eval()

def lyapincr(X, r, beta, gamma, pFeLi, model, likelihood):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        sigma = torch.sqrt(likelihood(model((X))).variance)
    return torch.sqrt(torch.sum((X - r) ** 2, dim=1)) <= (torch.sqrt(beta) * sigma + gamma) / (pFeLi.kc * torch.sqrt(torch.tensor(pFeLi.lam) ** 2 + 1))

def dyn(t, x):
    return dynAffine(t, x, ctrlFeLi, ref, pdyn_f, pdyn_g, pFeLi)

print('Simulating Trajectories...')

t = torch.linspace(0, Tsim, Nsim)
Xsim = odeint(dyn, x0.float(), t, method='rk4').squeeze(1)
with open('Xsim.txt', 'w') as f:
    f.write(str(Xsim))
Xd = ref(t).T
Xd = Xd[:, 0:E]
AreaError = [None]*Nsim
ihull = [None]*Nsim
for nsim in range(Nsim):
    ii = torch.where(lyapincr(Xte, Xd[nsim, 0:2], beta, gamma, pFeLi, model, likelihood))
    ii = ii[0]
    ihull[nsim] = ii[ConvexHull(Xte[ii, :]).vertices]
    shell = []
    for i in ihull[nsim]:
        shell.append((Xte[i, 0].item(), Xte[i, 1].item()))
    AreaError[nsim] = Polygon(shell).area
# print('AreaError: ', AreaError)
    
# ==================== #
print('Plotting Results...')
if_plot = True

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if if_plot:
    Nte = 1e4
    Ndte = int(Nte**(1/E))
    Nte = Ndte**E
    Xtes = ndgridj(XteMin, XteMax, Ndte*torch.ones((E, 1)))
    Xtes1 = Xte[:, 0].reshape(Ndte, Ndte)
    Xtes2 = Xte[:, 1].reshape(Ndte, Ndte)

    # Plot full trajectories
    fig, ax = plt.subplots()
    Z = (likelihood(model(Xtes)).variance - 1e5).reshape(Ndte, Ndte).T
    Z = Z.detach().numpy()
    c = ax.imshow(Z, cmap='cool', origin='lower', extent=[Xtes1.min(), Xtes1.max(), Xtes2.min(), Xtes2.max()])
    # fig.colorbar(c, ax=ax) # show colorbar
    ax.plot(Xsim[:, 0], Xsim[:, 1], 'b') # simulated trajectory
    ax.plot(Xd[:, 0], Xd[:, 1], 'r') # reference trajectory
    ax.plot(Xtr[:, 0], Xtr[:, 1], 'kx')
    ax.set_xlabel('x1') 
    ax.set_ylabel('x2')  
    plt.show()

    # Plot Tracking Error
    norme = torch.norm(Xsim - Xd, dim=1)
    plt.figure()
    plt.xlabel('t')
    plt.ylabel('|e| and |B|')
    plt.semilogy(t, norme, 'b')
    plt.semilogy(t, AreaError, 'r')
    plt.legend(['|e|', '|B|'])
    plt.show()

    # Plot f vs fh
    plt.figure()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()