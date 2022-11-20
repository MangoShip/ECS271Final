import os
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

# Loading and preprocessing images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224), # Each image has dimension of [3, 224, 224]
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet's mean and std

trainset = torchvision.datasets.ImageFolder('./datasets/Sohas_weapon-Classification', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4, shuffle = True)

classes = ('billete', 'knife', 'monedero', 'pistol', 'smartphone', 'tarjeta')

#displayImageDimensions('./datasets/Sohas_weapon-Classification')
