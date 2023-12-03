import torch
from gp_utils import *

def refGeneral(t, E, reffun):
    if t.dim() == 0: # if t is a tensor scalar or scalar
        t = t.unsqueeze(0)

    r = torch.zeros(E, t.shape[0])
    r[0, :] = reffun(t)
    for e in range(1, E):
        reffun = lambda x: gradestj(reffun, x)
        r[e, :] = reffun(t)
    return r