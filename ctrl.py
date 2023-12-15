import torch

def ctrlFeLi(t, x, p, reffun):
    E = x.shape[1]
    # print('t: ', t)
    xd = reffun(t)
    e = x - xd[0:E].T
    # print('x: ', x.shape)
    # print('xd: ', xd)
    # print('e: ', e)
    r = torch.matmul(torch.tensor([[p.lam, 1.0]]), e.T)
    # print('xd[E]: ', xd[E])
    rho = p.lam * e[:, 1:E] - xd[E]
    # print('rho: ', rho)
    nu = -p.kc * r - rho
    u = 1. / p.g(x) * (-p.f(x) + nu)
    return u
