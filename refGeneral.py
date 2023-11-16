import torch
from gp_utils import *

def refGeneral(t, E, reffun):
    if not torch.is_tensor(E) or E.dim() != 0 or not torch.is_tensor(t) or t.dim() != 1:
        raise ValueError('wrong input dimension')
    
    r = torch.zeros(E, t.shape[0])
    r[0, :] = reffun(t)
    for e in range(1, E):
        reffun = gradestj(reffun)
        r[e, :] = reffun(t)
    return r