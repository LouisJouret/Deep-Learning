import torch

def dMSE(v, t):
    dmse = 2 * (v - t)
    return dmse

def MSE(v, t):
    loss = torch.pow(v - t.view(-1, 1), 2)
    return loss

def d_cross_entropy(v, t):
    dce = (v-t)/(v*(1-v))
    return dce