import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from task4a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)

NUM_OUTPUTS = 10

def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float: 
    """
    Args:
        X: images of shape [batch size, 785]
        outputs: outputs of model of shape: [batch size, 10]
        targets: labels/targets of each image of shape: [batch size, 10]
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
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        l2_reg_lambda: float
        ):
    global X_train, X_val, X_test
    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}

    # Intialize our model
    model = SoftmaxModel(l2_reg_lambda)

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
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
    return model, train_loss, val_loss, train_accuracy, val_accuracy


# Load dataset
validation_percentage = 0.1
X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
    validation_percentage)

X_train = pre_process_images(X_train)
X_val = pre_process_images(X_val)
X_test = pre_process_images(X_test)

# One-hot encode our targets
Y_train = one_hot_encode(Y_train, NUM_OUTPUTS)
Y_val = one_hot_encode(Y_val, NUM_OUTPUTS)
Y_test = one_hot_encode(Y_test, NUM_OUTPUTS)

# Hyperparameters
num_epochs = 50
learning_rate = .3
batch_size = 128

# Training
model, train_loss, val_loss, train_accuracy, val_accuracy = train(
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    batch_size=batch_size,
    l2_reg_lambda=0)

print("Final Train Cross Entropy Loss:",
      cross_entropy_loss(Y_train, model.forward(X_train)))
print("Final Validation Cross Entropy Loss:",
      cross_entropy_loss(Y_test, model.forward(X_test)))
print("Final Test Cross Entropy Loss:",
      cross_entropy_loss(Y_val, model.forward(X_val)))

print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
print("Final Test accuracy:", calculate_accuracy(X_test, Y_test, model))


# Plot loss
plt.ylim([0.01, .2])
utils.plot_loss(train_loss, "Training Loss")
utils.plot_loss(val_loss, "Validation Loss")
plt.xlabel("Gradient steps")
plt.legend()
plt.savefig("softmax_train_loss.png")
plt.show()

# Plot accuracy
plt.ylim([0.8, .95])
utils.plot_loss(train_accuracy, "Training Accuracy")
utils.plot_loss(val_accuracy, "Validation Accuracy")
plt.xlabel("Gradient steps")
plt.legend()
plt.show()


lambda_values = [0, 0.1]
lambda_validation_accuracies = []
weight_lengths = []
weights_as_images = []

for l2_reg_lambda in lambda_values:
    # hyperparameters
    num_epochs = 50
    learning_rate = 0.2
    batch_size = 128
    
    # Train
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        l2_reg_lambda=l2_reg_lambda)

    # Print final losses for current lambda
    print("========== lambda:", str(l2_reg_lambda), "==========")
    print("Final Train Cross Entropy Loss:",
        cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
        cross_entropy_loss(Y_test, model.forward(X_test)))
    print("Final Test Cross Entropy Loss:",
        cross_entropy_loss(Y_val, model.forward(X_val)))

    # Print accuracy for current lambda
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    print("Test accuracy:", calculate_accuracy(X_test, Y_test, model))
    
    lambda_validation_accuracies.append(val_accuracy)
    length = model.w.transpose().dot(model.w)[0][0]
    weight_lengths.append(length)
    
    # Transform our weights into image matrices
    image = np.zeros((28,0))
    for i in range(NUM_OUTPUTS):
        digitImage = model.w[:-1, i].reshape(28, 28)
        image = np.hstack((image, digitImage))
        
    weights_as_images.append(image)

# Plot accuracies
plt.ylim([0.6, 0.95])
for (l2_reg_lambda, accuracies) in zip(lambda_values, lambda_validation_accuracies):
    utils.plot_loss(accuracies, str(l2_reg_lambda))
plt.legend(title="Lambda")
plt.savefig("binary_train_accuracy_regularization.png")
plt.show()

# Plot weight lengths
ind = np.arange(lambda_values.__len__())
plt.bar(ind, weight_lengths)
plt.xticks(ind, lambda_values)
plt.ylabel("Length")
plt.xlabel("Lambda")
plt.savefig("binary_train_weight_length.png")
plt.show()

# Show images
fig, ax = plt.subplots(lambda_values.__len__(), 1, subplot_kw={'xticks': [], 'yticks': []})
for (ax, img, l2_reg_lambda) in zip(ax.flat, weights_as_images, lambda_values):
    ax.imshow(img, cmap="gray")
    ax.set_title(str(l2_reg_lambda))
plt.savefig("binary_train_weight_image.png")
plt.show()


