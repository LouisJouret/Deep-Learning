import DL_framework as DL
import torch
import numpy as np
import matplotlib.pyplot as plt

def data_generate():
    # create 1000 points uniformely distributed
    train_input = torch.rand(2, 1000)
    test_input = torch.rand(2, 1000)

    # label the points
    train_target = torch.zeros(1,1000)
    test_target = torch.zeros(1,1000)
    radius = (1/np.sqrt(2*np.pi))

    # if a point is inside the circle it get the value of 1
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

    # check if a point is correctly labeled
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

    # check if correctly labeled
    for i in range(target.size(1)):
        if (result[i] > 0.5) and (target[:,i] == 0):
            nb_errors = nb_errors + 1
        elif (result[i] <= 0.5) and (target[:,i] == 1):
            nb_errors = nb_errors + 1

    # compute percentage
    error = (100 * nb_errors) / target.size(1)
    return error
def plot(*args):
    # plot the error or loss log depending on the number of inputs
    plt.clf()

    if (len(args)>1):
        plt.plot(args[0])
        plt.plot(args[1])
        plt.legend(["Train Error", "Test Error"], loc="upper right")
        plt.xlabel("Epochs")
        plt.ylabel("Error [%]")
    else:
        plt.plot(args[0])
        plt.legend(["Loss"], loc="upper right")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    plt.pause(0.05)
    plt.show()

plt.ion()

# generate the data
train_input, train_target, test_input, test_target = data_generate()

# determine the parameters
epsilon = 0.25
learning_rate = 0.0001
Epochs = 500
batch_size = 1

# create the model
myModel = DL.Sequential((2, 'relu'),
                        (25, 'relu'),
                        (25, 'relu'),
                        (25, 'relu'),
                        (1, 'tanh'),
                        epsilon)

# initialize the Log variables for plotting
TestErrorLog = []
TrainErrorLog = []
TrainLossLog = []

# training loop
for k in range(Epochs):
    # train the model with a loss function
    myModel, Loss = DL.train(myModel, train_input, batch_size, 'mse', train_target, learning_rate)

    # compute the errors
    test_err = error(myModel, test_input, test_target)
    train_err = error(myModel, train_input, train_target)

    # log the errors
    TrainLossLog.append(torch.mean(Loss))
    TestErrorLog.append(test_err)
    TrainErrorLog.append(train_err)

    # plot the Epoch's results
    plot(TrainErrorLog, TestErrorLog)
    print('Epoch', k + 1)
    print('Loss :', float(TrainLossLog[-1]))
    print(' Train Error: ', train_err)
    print(' Test Error: ', test_err)
    print('######################')


# final results
plot(TrainErrorLog, TestErrorLog)
plt.savefig('Error_plot.png')
plot(TrainLossLog)
plt.savefig('Loss_plot.png')
plot_final_result(myModel, test_input, test_target)