import torch
import gpytorch
import numpy as np
import math

# Set Parameters
Ntr = 100
Tsim = 30
Nsim = 200
sn = 0.1
E = 2

# Initial State / reference for simulation
x0 = torch.tensor([0.0, 0.0])
ref = lambda t: torch.tensor([2 * torch.sin(t)])

# Controller gains
lam = torch.ones(E - 1)
kc = 2.0

# Define System Dynamics
a = 1
b = 1
c = 0

def pdyn(x):
    f = 1 - torch.sin(x[:, 0]) + 1 / (1 + torch.exp(-x[:, 1]))
    g = 1
    return f, g

# GP learning and simulation parameters
optGPR = {
    'kernel': gpytorch.kernels.RBFKernel(),
    'likelihood': gpytorch.likelihoods.GaussianLikelihood(),
}
odeopt = {'rtol': 1e-3, 'atol': 1e-6}

# Visualization
Nte = 1e4
XteMin = torch.tensor([-6.0, -4.0])
XteMax = torch.tensor([4.0, 4.0])
Ndte = int(Nte**(1 / E))
Nte = Ndte**E

Xte = torch.meshgrid(
    torch.linspace(XteMin[0], XteMax[0], Ndte),
    torch.linspace(XteMin[1], XteMax[1], Ndte)
)

Xte = torch.cat([x.reshape(-1, 1) for x in Xte], dim=1)

Ntrajplot = 100

# Lyapunov test
tau = 1e-8
delta = 0.01
deltaL = 0.01

# Generate Training Points
Ntr = int(Ntr**(1 / E))**E
grid_x = torch.linspace(0, 3, int(math.sqrt(Ntr)))
grid_y = torch.linspace(-3, 3, int(math.sqrt(Ntr)))
Xtr = torch.meshgrid(grid_x, grid_y)
Xtr = torch.stack((Xtr[0].flatten(), Xtr[1].flatten()), axis=1)
Ytr = pdyn(Xtr)[0] + sn * torch.randn(Ntr)


gprModel = gpytorch.models.ExactGP(Xtr, Ytr, optGPR['likelihood'])
gprModel.covar_module = optGPR['kernel']
# gpytorch.settings.likelihood.confidence_min = 1e-4
# gpytorch.settings.likelihood.confidence_max = 1e4
# gpytorch.settings.likelihood.noise_constraint = gpytorch.constraints.GreaterThan(1e-4)


# Test Lyapunov condition
Lf = max(torch.norm(gpytorch.utils.gradient(0, pdyn(Xte)[0])**2, dim=0)**0.5)
Lk = torch.norm(optGPR['kernel'](Xte).evaluate()).item()

def k(x, xp):
    return (optGPR['likelihood'].noisy_train_labels.mean()**2) * torch.exp(
        -0.5 * torch.sum((x - xp)**2 / (optGPR['kernel'].lengthscale**2), dim=0)
    )

def dkdxi(x, xp, i):
    return -(x[i] - xp[i]) / (optGPR['kernel'].lengthscale[i]**2) * k(x, xp)

def ddkdxidxpi(x, xp, i):
    return (optGPR['kernel'].lengthscale[i]**(-2)) * k(x, xp) + (x[i] - xp[i]) / (optGPR['kernel'].lengthscale[i]**2) * dkdxi(x, xp, i)

def dddkdxidxpi(x, xp, i):
    return -(optGPR['kernel'].lengthscale[i]**(-2)) * dkdxi(x, xp, i) - (optGPR['kernel'].lengthscale[i]**(-2)) * dkdxi(x, xp, i) + (x[i] - xp[i]) / (optGPR['kernel'].lengthscale[i]**2) * ddkdxidxpi(x, xp, i)

r = torch.max(gpytorch.utils.pdist(Xte))
Lfs = torch.zeros(E)
for e in range(E):
    maxk = torch.max(ddkdxidxpi(Xte, Xte, e))
    Lkds = torch.zeros(Nte)
    for nte in range(Nte):
        Lkds[nte] = torch.max(dddkdxidxpi(Xte, Xte[:, nte], e))
    Lkd = torch.max(Lkds)
    Lfs[e] = torch.sqrt(2 * np.log(2 * E / deltaL)) * maxk + 12 * torch.sqrt(6 * E) * torch.max(maxk, torch.sqrt(r * Lkd))
Lfh = torch.norm(Lfs)
Lnu = Lk * torch.sqrt(Ntr) * torch.norm(likelihood.noise)
omega = torch.sqrt(2 * tau * Lk * (1 + Ntr * torch.norm(optGPR['kernel'](Xtr, Xtr) + sn**2 * torch.eye(Ntr)) * optGPR['likelihood'].noisy_train_labels.mean()**2))
beta = 2 * np.log((1 + ((torch.max(XteMax) - torch.min(XteMin)) / tau))**E / delta)
gamma = tau * (Lnu + Lfh) + torch.sqrt(beta) * omega

def Lyapincr(X, r):
    return torch.norm(X - r, dim=0) <= (torch.sqrt(beta) * sigfun(X) + gamma) / (kc * torch.sqrt(lam**2 + 1))

# Simulate System with Feedback Linearization and PD Controller
print('Simulation...')

def ctrlFeLi(t, x, pFeLi, ref):
    f, _ = pdyn(x)
    u = -pFeLi.kc * (x - ref(t))
    return f + pFeLi.lam * torch.tanh(u)

def dynAffine(t, x, ctrl, pdyn):
    f, _ = pdyn(t, x)
    u = ctrl(t, x)
    return f + u

T = torch.linspace(0, Tsim, Nsim)
Xsim = torch.zeros((E, Nsim))

for i in range(Nsim):
    t = T[i]
    x = Xsim[:, i]
    f, _ = pdyn(x)
    u = ctrlFeLi(t, x, {'kc': kc, 'lam': lam}, ref)
    Xsim[:, i + 1] = x + (f + u) * (T[i + 1] - T[i])

Xd = torch.stack([ref(t) for t in T])
AreaError = torch.zeros(Nsim)
ihull = [None] * Nsim

for nsim in range(Nsim):
    Xsim_np = Xsim.numpy()
    Xte_np = Xte.numpy()
    ii = np.where(Lyapincr(Xte_np, Xd[:2, nsim].numpy()))[0]
    ihull[nsim] = ii[ConvexHull(Xte_np[:, ii].T).vertices]
    xhull = Xte[:, ihull[nsim]].T
    AreaError[nsim] = ConvexHull(xhull).volume

# Visualization (You need to create a suitable visualization function here)
print('Plotting Results...')
   
