import torch
import torchvision
import torchvision.transforms as transforms

class DataLoader(object):
    def __init__(self):
        super(DataLoader, self).__init__()

        # Loading and preprocessing images
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), # Each image has dimension of [3, 224, 224]
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # ImageNet's mean and std

        self.trainset = torchvision.datasets.ImageFolder('./datasets/Sohas_weapon-Classification', transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size = 4, shuffle = True)

        self.classes = ('billete', 'knife', 'monedero', 'pistol', 'smartphone', 'tarjeta')