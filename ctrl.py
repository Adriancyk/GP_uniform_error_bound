import torch

def ctrlFeLi(t, x, p, reffun):
    E = x.shape[0]
    xd = reffun(t)
    e = x - xd[0:E]

    r = torch.matmul(torch.cat((p['lam'], torch.tensor([1]))), e)
    rho = torch.matmul(p['lam'], e[1:E]) - xd[E]
    nu = -p['kc'] * r - rho
    u = 1. / p['g'](x) * (-p['f'](x) + nu)
    return u