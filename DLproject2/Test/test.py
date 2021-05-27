import DL_framework as DL
import torch
import numpy as np
import matplotlib.pyplot as plt

def data_generate():
    train_input = torch.rand(2, 1000)
    test_input = torch.rand(2, 1000)
    train_target = torch.zeros(1,1000)
    test_target = torch.zeros(1,1000)
    radius = (1/np.sqrt(2*np.pi))

    for i in range(1000):
        if ((train_input[0,i]-0.5) ** 2 + (train_input[1,i]-0.5) ** 2 < radius ** 2):
            train_target[:,i] = 1.0
        if ((test_input[0,i]-0.5) ** 2 + (test_input[1,i]-0.5) ** 2 < radius ** 2):
            test_target[:,i] = 1.0

    return train_input, train_target, test_input, test_target
def plot_final_result(model,input,target):
    plt.clf()
    inside = []
    outside = []
    correct = []
    error = []

    res_model, _ = DL.forward(model, input)
    pred = res_model.x[res_model.nb_layers-1].flatten()
    for i in range(pred.size(0)):
        if pred[i] <= 0.5:
            outside.append(input[:,i].numpy())
            if target[:,i] == 0:
                correct.append(input[:,i].numpy())
            else:
                error.append(input[:,i].numpy())
        else:
            inside.append(input[:,i].numpy())
            if target[:,i] == 1:
                correct.append(input[:,i].numpy())
            else:
                error.append(input[:,i].numpy())

    inside = np.vstack(inside)
    outside = np.vstack(outside)
    plt.scatter(inside[:, 0], inside[:, 1])
    plt.scatter(outside[:, 0], outside[:, 1])
    plt.legend(["Inside", "Outside"], loc="upper right")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('square')
    plt.savefig('Result.png')
    correct = np.vstack(correct)
    error = np.vstack(error)
    plt.scatter(correct[:, 0], correct[:, 1])
    plt.scatter(error[:, 0], error[:, 1])
    plt.legend(["Correct", "False"], loc="upper right")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('square')
    plt.savefig('Error.png')
def error(model, input, target):
    myModel, _ = DL.forward(model, input)
    nb_errors = 0
    result = myModel.x[myModel.nb_layers - 1].flatten()
    for i in range(target.size(1)):
        if (result[i] > 0.5) and (target[:,i] == 0):
            nb_errors = nb_errors + 1
        elif (result[i] <= 0.5) and (target[:,i] == 1):
            nb_errors = nb_errors + 1
    error = (100 * nb_errors) / target.size(1)
    return error
def plot(TestErrorLog,TrainErrorLog):
    plt.clf()
    plt.plot(TrainErrorLog)
    plt.plot(TestErrorLog)
    plt.legend(["Train Error", "Test Error"], loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Error [%]")
    plt.pause(0.05)
    plt.show()

plt.ion()

train_input, train_target, test_input, test_target = data_generate()

epsilon = 0.2
learning_rate = 0.001
Epochs = 100
batch_size = 1

myModel = DL.Sequential((2, 'relu'),
                        (25, 'relu'),
                        (25, 'relu'),
                        (25, 'relu'),
                        (1, 'tanh'),
                        epsilon)
TestErrorLog = []
TrainErrorLog = []

for k in range(Epochs):
    myModel = DL.trainSGD(myModel, train_input, batch_size, 'mse', train_target, learning_rate)
    test_err = error(myModel, test_input, test_target)
    train_err = error(myModel, train_input, train_target)
    TestErrorLog.append(test_err)
    TrainErrorLog.append(train_err)
    plot(TestErrorLog, TrainErrorLog)
    print('Epoch', k + 1)
    print(' Train Error: ', train_err)
    print(' Test Error: ', test_err)
    print('######################')


plot(TestErrorLog, TrainErrorLog)
plt.savefig('Error_plot.png')
plot_final_result(myModel, train_input,train_target)