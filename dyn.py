import torch

def dynAffine(t, x, ctrl, p_f, p_g):
    E = x.shape[0]
    dxdt = torch.zeros(E)
    dxdt[0:E-1] = x[1:E]
    dxdt[E-1] = p_f(x) + p_g(x) * ctrl(t, x)
    return dxdt