from DL_framework.Activation import *
from DL_framework.Losses import *
from DL_framework.Update import *

class Sequential(object):

    def __init__(self, *args):
        # initializing all the dictionaries
        self.activation = {}
        self.dactivation = {}
        self.w = {}
        self.b = {}
        self.x = {}
        self.s = {}
        self.dl_dw = {}
        self.dl_db = {}
        self.dl_ds = {}
        self.layers = {}
        self.epsilon = args[-1]
        self.nb_layers = len(args) - 1

        # allocate the activation functions for every layer
        for i in range(self.nb_layers):
            self.layers[i] = args[i][0]
            if args[i][1] == 'tanh':
                self.activation[i] = tanh
                self.dactivation[i] = dtanh
            elif args[i][1] == 'relu':
                self.activation[i] = relu
                self.dactivation[i] = drelu
            elif args[i][1] == 'leaky_relu':
                self.activation[i] = leaky_relu
                self.dactivation[i] = dleaky_relu
            elif args[i][1] == 'softmax':
                self.activation[i] = softmax
                self.dactivation[i] = dsoftmax
            else:
                print("Activation function not supported")

        #initializing the weights, biases and their derivatives
        for i in range(self.nb_layers - 1):
            self.w[i] = torch.empty(self.layers[i + 1], self.layers[i]).normal_(0, self.epsilon)
            self.b[i] = torch.empty(self.layers[i + 1]).normal_(0, self.epsilon)
            self.dl_dw[i] = torch.zeros_like(self.w[i])
            self.dl_db[i] = torch.zeros_like(self.b[i])

def forward(model, input):
    # feeding the input data to the first layer
    model.x[0] = input
    # for every layer calculate the new s and x
    for i in range(model.nb_layers - 1):
        model.s[i] = model.w[i].mm(model.x[i]) + model.b[i].view(-1, 1)
        model.x[i + 1] = model.activation[i](model.s[i])
    # compute the prediction in case of multiple classes
    output = torch.argmax(model.x[model.nb_layers - 1], dim=0)
    return model, output

def backward(model, loss_function, target):
    # compute dl_ds depending on the loss function
    if (loss_function == 'mse'):
        model.dl_ds[model.nb_layers - 2] = model.dactivation[model.nb_layers - 1](model.s[model.nb_layers - 2]) * dMSE(model.x[model.nb_layers - 1], target)
    elif (loss_function == 'cross_entropy'):
        model.dl_ds[model.nb_layers - 2] = model.dactivation[model.nb_layers - 1](model.s[model.nb_layers - 2]) * d_cross_entropy(model.x[model.nb_layers - 1], target)

    # propagating the error backwards / computing dl_ds for every layer
    for i in reversed(range(model.nb_layers - 2)):
        model.dl_ds[i] = model.dactivation[i](model.s[i]) * model.w[i + 1].t().mm(model.dl_ds[i + 1])

    # compute dl_dw and dl_ds for every layer
    for i in range(model.nb_layers - 1):
        model.dl_dw[i] = (model.dl_ds[i].mm(torch.transpose(model.x[i], 0, 1)))
        model.dl_db[i] = torch.sum(model.dl_ds[i], 1)
    return model

def train(model, input, batch_size, loss_function, target, lr):
    model_train = model
    # permute the data in case it is not shuffled yet
    indices = torch.randperm(target.size(1))
    target = target[:, indices]
    input = input[:, indices]

    # initialize the loss
    Loss = torch.zeros_like(target)

    # loop in which we feed every batch to the network
    for i in range(int(input.size(1) / batch_size)):

        # fill the batches
        batch = input[:, i * batch_size: i * batch_size + batch_size]
        target_batch = target[:, i * batch_size: i * batch_size + batch_size]

        # feed the network with a batch
        model_train, _ = forward(model_train, batch)

        # compute the loss in terms of the loss function
        if loss_function == 'mse':
            Loss[:,i : i + batch_size] = MSE(model_train.x[model_train.nb_layers - 1], target_batch)
        else:
            Loss[:, i: i + batch_size] = cross_entropy(model_train.x[model_train.nb_layers - 1], target_batch)

        # compute the backpropagation
        model_train = backward(model_train, loss_function, target_batch)

        # update the model
        model_train = update(model_train, lr)
    return model_train, Loss
