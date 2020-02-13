import numpy as np
import utils
import typing
from math import sqrt

np.random.seed(1)

meanPixelValue = None
meanPixelDeviation = None

meanPixelValue_pixel_normalization = None
meanPixelDeviation_pixel_normalization = None

def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    
    # Normalize
    X = (X - meanPixelValue) / meanPixelDeviation

    # Apply bias trick
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((X, ones),axis=1)

def pre_process_images_pixel_normalization(X: np.ndarray, epsilon=1e-5):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785]
    """
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
    
    # Normalize
    X = (X - meanPixelValue_pixel_normalization) / (meanPixelDeviation_pixel_normalization + epsilon)

    # Bias trick
    ones = np.ones((X.shape[0], 1))
    return np.concatenate((X, ones),axis=1)

def calc_mean_and_deviation(X: np.array):
    global meanPixelValue, meanPixelDeviation
    meanPixelValue = X.mean()
    meanPixelDeviation = X.std()

def calc_mean_and_deviation_pixel_normalization(X: np.array):
    global meanPixelValue_pixel_normalization, meanPixelDeviation_pixel_normalization
    meanPixelValue_pixel_normalization = X.mean(axis=0)
    meanPixelDeviation_pixel_normalization = X.std(axis=0)

def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    
    ce = targets * np.log(outputs)    
    N = targets.shape[0]
    return -sum(sum(ce)) / N
    
## Activation functions and derivatives

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    expOfMinusX = np.exp(-x)
    return expOfMinusX / np.power(1 + expOfMinusX, 2)

def improved_sigmoid(x):
    return 1.7159 * np.tanh(2/3 * x)

def improved_sigmoid_derivative(x):
    return 2/3 * 1.7159 * (1 - np.tanh(2/3 * x)**2)

def softmax(z):
    exp_z  = np.exp(z.transpose())
    return (exp_z / sum(exp_z)).transpose()



class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        self.sigmoid = lambda x: improved_sigmoid(x) if use_improved_sigmoid else sigmoid(x)
        self.sigmoid_derivative = lambda x: improved_sigmoid_derivative(x) if use_improved_sigmoid else sigmoidDerivative(x)

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []
        prev = self.I
        for size in self.neurons_per_layer:
            w_shape = (prev, size)
            print("Initializing weight to shape:", w_shape)

            if use_improved_weight_init:
                sigma = 1 / np.sqrt(prev)
                w = np.random.normal(0, sigma, size=w_shape)
            else:
                w = np.random.uniform(-1, 1, size=w_shape)
            
            self.ws.append(w)
            prev = size
        self.grads = [None for i in range(len(self.ws))]
        
        # Buffers for storing activations and weighted sums of the previous iteration (needed for backprop)
        self.aBuffer = None
        self.zBuffer = None


    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        
        self.aBuffer = []
        self.zBuffer = []
    
        for l in range(len(self.ws)):
            prevActivation = X if l == 0 else self.aBuffer[-1]
            activationFunc = self.sigmoid if l < len(self.ws) - 1 else softmax
            self.zBuffer.append(prevActivation.dot(self.ws[l]))
            self.aBuffer.append(activationFunc(self.zBuffer[-1]))
    
        return self.aBuffer[-1]    
        
    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        assert targets.shape == outputs.shape,\
            f"Output shape : {outputs.shape}, targets: {targets.shape}"
        # A list of gradients.
        # For example, self.grads[0] will be the gradient for the first hidden layer
        self.grads = []

        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
        
        N = X.shape[0]
        
        deltas = [0 for _ in self.ws]
        
        # Output error
        deltas[-1] = -(targets - outputs) / N
        # Back-propagation
        for l in reversed(range(len(self.ws) - 1)):
            deltas[l] = deltas[l+1].dot(self.ws[l+1].transpose()) * self.sigmoid_derivative(self.zBuffer[l])
        # Update gradients
        for l in range(len(self.ws)):
            activation = X if l == 0 else self.aBuffer[l-1]
            self.grads.append(activation.transpose().dot(deltas[l]))            

    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]


def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    encoded = np.zeros((Y.shape[0], num_classes))
    for i in range(Y.shape[0]):
        encoded[i,Y[i,0]] = 1
    return encoded

def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist(0.1)
    calc_mean_and_deviation(X_train)
    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)
