def update(model,lr):
    # for every loop update the weights and biases using the derivatives
    for i in range(model.nb_layers - 1):
        model.w[i] = model.w[i] - lr * model.dl_dw[i]
        model.b[i] = model.b[i] - lr * model.dl_db[i]
        # clear the derivatives
        model.dl_dw[i].zero_()
        model.dl_db[i].zero_()
    return model

