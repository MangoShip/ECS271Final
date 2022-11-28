import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import SubsetRandomSampler


class DataLoader(object):
    def __init__(self):
        super(DataLoader, self).__init__()

        # Make all random sequences on all computers the same.
        np.random.seed(1)

        self.classes = ('billete', 'knife', 'monedero', 'pistol', 'smartphone', 'tarjeta')
        
        # Loading and preprocessing images
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # Each image has dimension of [3, 224, 224]
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet's mean and std

        self.dataset = torchvision.datasets.ImageFolder('./datasets/Sohas_weapon-Classification', transform=transform)

        # Perform balanced split training-test dataset
        classes_size = [777, 2349, 813, 3975, 1184, 446] # Number of files for each class
        classes_train_indices = []
        classes_test_indices = []
        test_ratio = 0.2 # How much in dataset will be test dataset
        indices = list(range(len(self.dataset)))
        start_index = 0
        
        for class_size in classes_size:
            # Get subset of indices for current class then shuffle
            class_indices = indices[start_index:(start_index + class_size)]
            np.random.shuffle(class_indices)
            
            test_index = int(test_ratio * len(class_indices))
            class_test_indices = class_indices[0:test_index]
            class_train_indices = class_indices[test_index:len(class_indices)]

            classes_test_indices += class_test_indices
            classes_train_indices += class_train_indices
            
            # Increment start_index for next class
            start_index += class_size

        train_sampler = SubsetRandomSampler(classes_train_indices)
        test_sampler = SubsetRandomSampler(classes_test_indices)

        # Load train and test dataset with samplers
        self.trainloader = torch.utils.data.DataLoader(self.dataset, batch_size = 4, sampler = train_sampler)
        self.testloader = torch.utils.data.DataLoader(self.dataset, batch_size = 4, sampler = test_sampler)
