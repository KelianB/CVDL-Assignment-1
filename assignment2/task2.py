import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
from task2a import calc_mean_and_deviation
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    outputs = model.forward(X)
    N = targets.shape[0]
    correctOutputs = 0
    
    for i in range(N):
        target = np.where(targets[i] == 1)[0][0]
        output = np.argmax(outputs[i])
        if target == output:
            correctOutputs += 1
    return correctOutputs / N


def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets
    #Moved to model
    #for layer in range(len(model.ws)):
    #    model.ws[layer] = np.random.uniform(-1, 1, size =model.ws[layer].shape)

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    #Variables for early stopping
    last_val_loss = 1
    best_val_loss = 1
    best_weights = None
    increased_last_time=False

    # Store last weights update term for momentum
    last_weights_update = []
    for l in range(len(model.ws)):
        last_weights_update.append(np.zeros_like(model.ws[l]))

    global_step = 0
    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            train_output = model.forward(X_batch)

            model.backward(X_batch, train_output, Y_batch)
            
            for l in range(len(model.ws)):
                if use_momentum:
                    update_term = momentum_gamma * last_weights_update[l] - learning_rate * model.grads[l]
                    model.ws[l] += update_term
                    last_weights_update[l] = update_term
                else:
                    model.ws[l] -= learning_rate * model.grads[l]    

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            if (global_step % num_steps_per_val) == 0:
                val_output = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, val_output)
                val_loss[global_step] = _val_loss
                
                train_output = model.forward(X_train)
                _train_loss = cross_entropy_loss(Y_train, train_output)
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
            
        # Shuffle results in accuracy of about 10%, need to look into that
        # In order to keep labels in the right order, we shuffle an array of indices
        # and then apply this ordering to both inputs and labels
        if use_shuffle:
            indices = np.arange(X_train.shape[0]);
            np.random.shuffle(indices)
            X_train = X_train[indices]
            Y_train = Y_train[indices]
            
        # Compute validation loss for early stopping
        val_outputs = model.forward(X_val)
        _val_loss = cross_entropy_loss(Y_val, val_outputs)
        if _val_loss <= best_val_loss:
            best_weights = model.ws
            best_val_loss = _val_loss
        if _val_loss > last_val_loss:
            model.ws = best_weights
            if increased_last_time:
                model.ws = best_weights
                break
            else:
                increased_last_time = True
        else:
            increased_last_time = False
        last_val_loss = _val_loss
            
    return model, train_loss, val_loss, train_accuracy, val_accuracy

def train_and_evaluate(
        neurons_per_layer: int,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_improved_sigmoid: bool,
        use_improved_weight_init: bool,
        use_momentum: bool,
        momentum_gamma: float):

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        datasets,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)
        
    print("----------", use_shuffle, use_improved_sigmoid, use_improved_weight_init, use_momentum, momentum_gamma, "----------")
    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))
    return train_loss, val_loss, train_accuracy, val_accuracy         
    

if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    # Pre-processing
    calc_mean_and_deviation(X_train)
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    X_test = pre_process_images(X_test)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    Y_test = one_hot_encode(Y_test, 10)

    # Hyperparameters
    num_epochs = 30
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    ## Task 2
    '''train_loss, val_loss, train_accuracy, val_accuracy = train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=False,
        use_improved_sigmoid=False,
        use_improved_weight_init=False,
        use_momentum=False,
        momentum_gamma=momentum_gamma)

    # Plot loss
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.ylim([0.1, .5])
    utils.plot_loss(train_loss, "Training Loss")
    utils.plot_loss(val_loss, "Validation Loss")
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    # Plot accuracy
    plt.ylim([0.9, 1.0])
    utils.plot_loss(train_accuracy, "Training Accuracy")
    utils.plot_loss(val_accuracy, "Validation Accuracy")
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.savefig("softmax_train_graph.png")
    plt.show()'''

    ## Task 3
    """
    results = []
    legend = ['Standard', 'Shuffle', 'Sigmoid', 'Init', 'Momentum']
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=False,
        use_improved_sigmoid=False,
        use_improved_weight_init=False,
        use_momentum=False,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=False,
        use_improved_weight_init=False,
        use_momentum=False,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=False,
        use_momentum=False,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=False,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=.02,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        momentum_gamma=.9))
    """
    
    ## Task 4a and 4b
    """
    results = []
    legend = ['Standard (64 hidden neurons)', '16 hidden neurons', '128 hidden neurons']
    learning_rate = 0.02
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        [16, 10],
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        [128, 10],
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        momentum_gamma=.9))
    """
    
    ## Task 4d
    results = []
    legend = ['Standard (64 hidden neurons)', '60 + 60 hidden neurons']
    learning_rate = 0.02
    results.append(train_and_evaluate(
        neurons_per_layer,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        momentum_gamma=.9))
    results.append(train_and_evaluate(
        [60, 60, 10],
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=True,
        use_improved_sigmoid=True,
        use_improved_weight_init=True,
        use_momentum=True,
        momentum_gamma=.9))
    
    ## Plotting for tasks 3 and 4
    # Plot loss
    plt.figure(figsize=(20, 16))
    plt.subplot(2, 2, 1)
    plt.ylim([0.0, .5])
    for (train_loss, val_loss, _, _), name in zip(results, legend):
        utils.plot_loss(train_loss, name)
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.subplot(2, 2, 3)

    plt.ylim([0.0, .5])
    for (train_loss, val_loss, _, _), name in zip(results, legend):
        utils.plot_loss(val_loss, name)
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Validation Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.ylim([0.86, 1.0])
    for (_, _, train_accuracy, val_accuracy), name in zip(results, legend):
        utils.plot_loss(train_accuracy, name)
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")

    plt.subplot(2, 2, 4)
    plt.ylim([0.86, 1.0])
    for (_, _, train_accuracy, val_accuracy), name in zip(results, legend):
        utils.plot_loss(val_accuracy, name)
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.savefig("softmax_train_variations_graph.png")
    plt.show()