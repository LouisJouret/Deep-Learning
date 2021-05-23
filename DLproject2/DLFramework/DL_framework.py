from Activation import *
from Losses import *

class DL(object):

    class Sequential (object):

        def __init__(self,*args):
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

            for i in range(self.nb_layers):
                self.layers[i] = args[i][0]
                if args[i][1] == 'tanh':
                    self.activation[i] = tanh
                    self.dactivation[i] = dtanh
                elif args[i][1] == 'relu':
                    self.activation[i] = relu
                    self.dactivation[i] = drelu
                elif args[i][1] == 'leaky_relu':
                    self.activation[i] = DL.leaky_relu
                    self.dactivation[i] = DL.dleaky_relu
                elif args[i][1] == 'softmax':
                    self.activation[i] = DL.softmax
                    self.dactivation[i] = DL.dsoftmax
                else:
                    print("Activation function not supported")

            for i in range(self.nb_layers-1):
                self.w[i] = torch.empty(self.layers[i+1], self.layers[i]).normal_(0, self.epsilon)
                self.b[i] = torch.empty(self.layers[i+1]).normal_(0, self.epsilon)
                self.dl_dw[i] = torch.empty(self.w[i].size())
                self.dl_db[i] = torch.empty(self.b[i].size())

    def forward(model, input):
        model.x[0] = input
        for i in range(model.nb_layers - 1):
            model.s[i] = model.w[i].mm(model.x[i]) + model.b[i].view(-1,1)
            model.x[i+1] = model.activation[i](model.s[i])
        model.x[model.nb_layers - 1] = model.activation[model.nb_layers-1](model.x[model.nb_layers - 1])
        output = torch.argmax(model.x[model.nb_layers-1],dim = 0)
        return model, output

    def backward(model,loss_function,target):
        if (loss_function == 'mse'):
            model.dl_ds[model.nb_layers - 2] = model.dactivation[model.nb_layers-1](model.s[model.nb_layers - 2]) * dMSE(model.x[model.nb_layers - 1], target)
        elif (loss_function == 'cross_entropy'):
            model.dl_ds[model.nb_layers - 2] = model.dactivation[model.nb_layers - 1](model.s[model.nb_layers - 2]) * d_cross_entropy(model.x[model.nb_layers - 1], target)
        for i in reversed(range(model.nb_layers-2)):
            model.dl_ds[i] = model.dactivation[i](model.s[i]) * model.w[i + 1].t().mm(model.dl_ds[i + 1])
        for i in range(model.nb_layers - 1):
            model.dl_dw[i] = model.dl_ds[i].mm(torch.transpose(model.x[i], 0, 1))
            model.dl_db[i] = torch.flatten(model.dl_ds[i])
        return model

    def train(model, loss_function, target, lr):
        model_after_back = DL.backward(model, loss_function, target)
        for i in range(model_after_back.nb_layers - 1):
            model_after_back.w[i] = model_after_back.w[i] - lr * model_after_back.dl_dw[i]
            model_after_back.b[i] = model_after_back.b[i] - lr * model_after_back.dl_db[i]
            model_after_back.dl_dw[i].zero_()
            model_after_back.dl_db[i].zero_()
        return model_after_back


