import torch
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math
from keras.datasets import mnist

torch.set_grad_enabled(False)

def tanh(x):
    sigma = torch.tanh(x)
    return sigma

def dtanh(x):
    dsigma = 1 - torch.pow(tanh(x), 2)
    return dsigma

def relu(x):
    x = torch.cat((x.reshape(x.size(0),1),torch.zeros(x.size(0),1)),1)
    relu, _ = torch.max(x,1)
    relu = relu.flatten()
    return relu

def drelu(x):
    drelu = torch.zeros_like(x)
    drelu[x > 0] = 1
    return drelu

def loss(v, t):
    loss = torch.pow(v - t, 2)
    return loss

def softmax(X):
    exps = torch.exp(X - torch.max(X))
    return exps / torch.sum(exps)

def cross_entropy(v, t):
    loss = torch.zeros(t.size(0))
    for i in range(t.size(0)):
        if t[i] == 1:
            loss[i] = - torch.log(v[i])
        else:
            loss[i] = - torch.log(1.0 - v[i])
    return loss

def d_cross_entropy(v,t):
    m = t.shape[0]
    dloss = softmax(v)
    dloss = dloss - t
    dloss = dloss / m
    return dloss

def dloss(v, t):
    dloss = 2 * (v - t)
    return dloss

def forward_pass(w, b, input,nb_layer):
    s = {}
    x = {}
    x[0] = input
    for i in range(nb_layer+1):
        s[i] = w[i].mv(x[i]) + b[i]
        x[i+1] = tanh(s[i])
    x[nb_layer+1] = softmax(x[nb_layer+1])
    return x, s

def backward_pass(w, x, s, t, dl_dw, dl_db, nb_layer):

    dl_ds[nb_layer] = dtanh(s[nb_layer]) * dloss(x[nb_layer+1],t)
    for i in reversed(range(nb_layer)):
        dl_ds[i] = dtanh(s[i]) * w[i+1].t().mv(dl_ds[i+1])
    for i in range(nb_layer+1):
        dl_dw[i] = (dl_ds[i].view(-1, 1).mm(x[i].view(1, -1)))
        dl_db[i] = (dl_ds[i])
    return dl_dw, dl_db

def convert_to_one_hot_labels(x):
    x = torch.tensor(x, dtype = torch.int32)
    nb_classes = torch.unique(x).size(0)
    one_hot = torch.zeros(nb_classes, x.size(0))
    for i in range(x.size(0)):
        one_hot[x.numpy()[i], i] = 1.0
    return nb_classes, one_hot

def load_mnist(size):
    (train_input, train_target), (test_input, test_target) = mnist.load_data()

    train_input = torch.tensor(train_input).float()
    train_target = torch.tensor(train_target)
    test_input = torch.tensor(test_input).float()
    test_target = torch.tensor(test_target)

    mu, std = train_input.mean(), train_input.std()
    train_input.sub_(mu).div_(std)
    test_input.sub_(mu).div_(std)

    train_input = train_input.narrow(0, 0, size)
    train_target = train_target.narrow(0, 0, size)
    test_input = test_input.narrow(0, 0, size)
    test_target = test_target.narrow(0, 0, size)

    # flatten
    train_input = train_input.clone().reshape(train_input.size(0), -1)
    test_input = test_input.clone().reshape(test_input.size(0), -1)

    return train_input, train_target, test_input, test_target

def data_generate():
    train_input = torch.rand(1000, 2)
    test_input = torch.rand(1000, 2)
    train_target = torch.zeros(1000)
    test_target = torch.zeros(1000)
    radius = (1/np.sqrt(2*np.pi))

    for i in range(1000):
        if ((train_input[i,0]-0.5) ** 2 + (train_input[i,1]-0.5) ** 2 < radius ** 2):
            train_target[i] = 1

    for i in range(1000):
        if ((test_input[i, 0]-0.5) ** 2 + (test_input[i, 1]-0.5) ** 2 < radius ** 2):
            test_target[i] = 1


    return train_input, train_target, test_input, test_target

def plot_result():
    plt.clf()
    inside = []
    outside = []
    for n in range(1000):
        x, _ = forward_pass(w, b, test_input[n], nb_layer)
        pred = x[nb_layer + 1].max(0)[1].item()
        if pred == 1:
            inside.append(test_input[n].numpy())
        else:
            outside.append(test_input[n].numpy())

    inside = np.vstack(inside)
    outside = np.vstack(outside)
    plt.scatter(inside[:,0],inside[:,1])
    plt.scatter(outside[:,0],outside[:,1])
    plt.axis('square')
    plt.savefig('Result.png')
    return


train_input, train_target, test_input, test_target = data_generate()
print(train_input)
print(train_target)
test_compare = test_target
train_compare = train_target
nb_classes, train_target = convert_to_one_hot_labels(train_target)
_ , test_target = convert_to_one_hot_labels(test_target)
print(train_target)

plt.ion()




Error_Plot_Train = []
Error_Plot_Test = []
Loss_Plot_Train = []
Loss_Plot_Test = []

nb_train_samples = train_input.size(0)

zeta = 1.0

train_target = train_target * zeta
test_target = test_target * zeta

nb_layer = 3
nb_hidden ={}
nb_hidden[0] = 25
nb_hidden[1] = 25
nb_hidden[2] = 25



eta = 0.0005

epsilon = 0.1


w = {}
b = {}
dl_dw = {}
dl_db = {}
dl_ds = {}

w[0] = torch.empty(nb_hidden[0], train_input.size(1)).normal_(0, epsilon)
b[0] = torch.empty(nb_hidden[0]).normal_(0, epsilon)

for i in range(nb_layer-1):
    w[i+1] = torch.empty(nb_hidden[i+1], nb_hidden[i]).normal_(0, epsilon)
    b[i+1] = torch.empty(nb_hidden[i+1]).normal_(0, epsilon)

w[nb_layer] = torch.empty(nb_classes, nb_hidden[nb_layer-1]).normal_(0, epsilon)
b[nb_layer] = torch.empty(nb_classes).normal_(0, epsilon)


for i in range(nb_layer+1):
        dl_dw[i] = torch.empty(w[i].size())
        dl_db[i] = torch.empty(b[i].size())

for k in range(5000):

    nb_train_errors = 0


    for n in range(nb_train_samples):
        x, s = forward_pass(w, b, train_input[n,:],nb_layer)
        pred = x[nb_layer+1].max(0)[1].item()
        #print('target',n,train_compare[n])
        #print('pred',n, train_target[pred, n])
        if (train_target[pred, n] < 0.5):
            nb_train_errors = nb_train_errors + 1
        dl_dw, dl_db = backward_pass(w, x, s, train_target[:, n], dl_dw, dl_db, nb_layer)
        # Gradient step
        for i in range(nb_layer + 1):
            w[i] = w[i] - eta * dl_dw[i]
            b[i] = b[i] - eta * dl_db[i]
            dl_dw[i].zero_()
            dl_db[i].zero_()
        Loss_Plot_Train.append(torch.mean(loss(x[nb_layer+1],train_target[:, n])).numpy())

    # Test error


    nb_test_errors = 0

    for n in range(test_input.size(0)):
        x, _ = forward_pass(w, b, test_input[n], nb_layer)
        pred = x[nb_layer+1].max(0)[1].item()
        if test_target[pred,n] < 0.5:
            nb_test_errors = nb_test_errors + 1
        Loss_Plot_Test.append(torch.mean(loss(x[nb_layer+1], test_target[:, n])).numpy())


    Loss_Plot_Train = np.mean(Loss_Plot_Train)
    Loss_Plot_Test= np.mean(Loss_Plot_Test)
    print('Epoch ', k + 1 ,' Train Loss: ',Loss_Plot_Train,' Test Loss: ', Loss_Plot_Test)
    print('Epoch ', k + 1, ' Train Error: ', (100 * nb_train_errors) / train_input.size(0), ' Test Error: ', (100 * nb_test_errors) / test_input.size(0))
    Error_Plot_Train.append((100 * nb_train_errors) / train_input.size(0))
    Error_Plot_Test.append((100 * nb_test_errors) / test_input.size(0))

    plt.clf()
    plt.plot(Error_Plot_Train)
    plt.plot(Error_Plot_Test)
    plt.legend(["Train Error", "Test Error"], loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Error [%]")
    #plt.ylabel("Loss")
    plt.pause(0.05)
    plt.show()
    Loss_Plot_Train = []
    Loss_Plot_Test = []

plt.savefig('Mnist.png')
plot_result()