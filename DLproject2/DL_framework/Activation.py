import torch

def tanh(x):
    tanh = torch.tanh(x)
    return tanh

def dtanh(x):
    dtanh = 1 - torch.pow(tanh(x), 2)
    return dtanh

def softmax(X):
    exps = torch.exp(X)
    exps = exps / torch.sum(exps)
    return exps

def dsoftmax(X):
    exps = torch.exp(X)
    dsoftmax = (exps * torch.sum(exps) - torch.pow(exps, 2)) / torch.pow(torch.sum(exps), 2)
    return dsoftmax

def relu(x):
    relu = torch.maximum(x, torch.zeros_like(x))
    return relu

def drelu(x):
    x[x > 0] = 1
    x[x < 0] = 0
    return x

def leaky_relu(x):
    relu = torch.maximum(x, 0.01*x)
    return relu

def dleaky_relu(x):
    x[x > 0] = 1
    x[x < 0] = 0.01
    return x

