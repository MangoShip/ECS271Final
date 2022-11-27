import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# Plot dimensions of all the images
def displayImageDimensions(path):
    imageTransform = transforms.Compose([transforms.ToTensor()])
    imageset = torchvision.datasets.ImageFolder(path, transform = imageTransform)

    image_heights = np.empty(len(imageset))
    image_widths = np.empty(len(imageset))

    for i in range(len(imageset)):
        image_shape = imageset[i][0].shape
        image_heights[i] = image_shape[1]
        image_widths[i] = image_shape[2]

    print("Average Width: " + str(np.mean(image_widths)))
    print("Average Height: " + str(np.mean(image_heights)))

    plt.scatter(image_widths, image_heights, color = 'blue', alpha = 0.5)
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.show()