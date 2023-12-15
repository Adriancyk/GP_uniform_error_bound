import torch
from gp_utils import *

def refGeneral(t, E, l_tau):
    if t.dim() == 0: # if t is a tensor scalar or scalar
        t = t.unsqueeze(0)
    r = torch.zeros(E, t.shape[0])
    r[0, :] = l_tau(t)

    for e in range(1, E):
        r[e, :] = gradestj1(gradestj1, t, e)
        # print('r: ', r[e,:])
    return r

if __name__ == '__main__':
    def reffun(t):
        return 2*torch.sin(t)
    E = 2
    t = torch.tensor([0.0])
    refGeneral(t, E+1, reffun)