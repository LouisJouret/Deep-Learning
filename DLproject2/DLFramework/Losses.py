import torch

def dMSE(v, t):
    dmse = 2 * (v - t)

    return dmse

def MSE(v, t):
    loss = torch.pow(v - t.view(-1, 1), 2)
    return loss

def d_cross_entropy(v, t):
    dce = torch.zeros_like(v)
    dce[0] = -t[0]/v[0] + (1-t[0])/(1-v[0])
    #dce[1] = -t[1] / v[1] + (1 - t[1]) / (1 - v[1])
    return dce