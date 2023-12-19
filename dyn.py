import torch

def dynAffine(t, x, ctrl, ref, p_f, p_g, pFeLi):
    # t: N
    # x: 1 x E
    E = x.shape[1]
    dxdt = torch.zeros(E).unsqueeze(0)
    dxdt[0, 0:E-1] = x[0, 1:E]
    # print('t: ', t)
    dxdt[0, E-1] = p_f(x) + p_g(x) * ctrl(t, x, pFeLi, ref)
    return dxdt