import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import cross_entropy_loss, BinaryModel, pre_process_images
np.random.seed(0)

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: BinaryModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        outputs: outputs of model of shape: [batch size, 1]
        targets: labels/targets of each image of shape: [batch size, 1]
    Returns:
        Accuracy (float)
    """
        
    # Task 2c
    outputs = model.forward(X)
    N = targets.shape[0]
    binary_threshold = 0.5
    correctOutputs = 0
    for i in range(N):
        if (targets[i] == 1 and outputs[i] >= binary_threshold) or (targets[i] == 0 and outputs[i] < binary_threshold):
            correctOutputs += 1
    return correctOutputs / N

def train(
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float
        ):
    """
        Function that implements logistic regression through mini-batch
        gradient descent for the given hyperparameters
    """
    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    model = BinaryModel(l2_reg_lambda)

    global_step = 0
    last_val_loss = 1
    best_val_loss = 1
    best_weights = None
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            # Select our mini-batch of images / labels
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # Forward pass
            train_outputs = model.forward(X_batch)
            
            # Backward propagation
            model.backward(X_batch, train_outputs, Y_batch)
            model.w -= learning_rate * model.grad

            # Track training loss continuously
            _train_loss = cross_entropy_loss(Y_batch, train_outputs)
            train_loss[global_step] = _train_loss

            # Track validation loss / accuracy every time we progress 20% through the dataset
            if global_step % num_steps_per_val == 0:
                val_outputs = model.forward(X_val)
                _val_loss = cross_entropy_loss(Y_val, val_outputs)
                val_loss[global_step] = _val_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1

        # Compute validation loss for early stopping
        val_outputs = model.forward(X_val)
        _val_loss = cross_entropy_loss(Y_val, val_outputs)
        if _val_loss <= best_val_loss:
            best_weights = model.w
            best_val_loss = _val_loss
        if _val_loss > last_val_loss:
            model.w = best_weights
            break
        last_val_loss = _val_loss
    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset (X_ = image data, Y_ = targets)
category1, category2 = 2, 3
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_binary_dataset(
    category1, category2, validation_percentage)

X_train = pre_process_images(X_train)
X_val = pre_process_images(X_val)
X_test = pre_process_images(X_test)

# Hyperparameters
num_epochs = 50
learning_rate = 0.2
batch_size = 128
l2_reg_lambda = 0

# Train
model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=l2_reg_lambda)

# Print final loss
print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Test Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))

# Print final accuracy
print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))

# Plot loss
plt.ylim([0., .4]) 
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.xlabel("Gradient steps")
plt.legend()
plt.savefig("binary_train_loss.png")
plt.show()

# Plot accuracy
plt.ylim([0.93, .99])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.xlabel("Gradient steps")
plt.legend()
plt.savefig("binary_train_accuracy.png")
plt.show()

