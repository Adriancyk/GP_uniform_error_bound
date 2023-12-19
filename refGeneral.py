import torch
from gp_utils import *

def refGeneral(t, E, ref):
    if t.dim() == 0: # if t is a tensor scalar or scalar
        t = t.unsqueeze(0)
    r = torch.zeros(E, t.shape[0])
    r[0, :] = ref(t)

    for e in range(1, E):
        r[e, :] = gradestj1(gradestj1, t, e)
        # print('r: ', r[e,:])
    return r

if __name__ == '__main__':
    def reffun(t):
        return 2*torch.sin(t)
    E = 2
    Tsim = 30
    Nsim = 200
    t = torch.linspace(0, Tsim, Nsim)
    refGeneral(t, E+1, reffun)