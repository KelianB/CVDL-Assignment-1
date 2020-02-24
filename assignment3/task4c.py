import matplotlib.pyplot as plt
import torchvision
from task4b import load_zebra_image, torch_image_to_numpy


if __name__ == "__main__":
    model = torchvision.models.resnet18(pretrained=True)
    layers = list(model.children())
    
    # Do a forward pass through all layers except the last two, and save activations
    activations = [load_zebra_image()]
    for i in range(len(layers) - 2):
        activation = layers[i](activations[-1])
        activations.append(activation)
    
    print("Last convolutional layer:", layers[i])
    
    last_activation = activations[-1]
    print("Last convolutional layer activation shape:", last_activation.shape)

    # Get the activation image for the first 10 filters of the last layer
    indices = list(range(10))
    activation_images = [torch_image_to_numpy(last_activation[0][i]) for i in indices]

    # Show images
    fig, ax = plt.subplots(2, len(indices)//2, subplot_kw={'xticks': [], 'yticks': []})
    for i in range(len(indices)):
        ax.flat[i].set_title(str(indices[i]))
        ax.flat[i].imshow(activation_images[i], cmap="gray")
        
    plt.savefig("last_layer_activations.png")
    plt.show()
