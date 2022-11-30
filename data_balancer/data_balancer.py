import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image

dataset_loc = "../datasets/Sohas_weapon-Classification"
new_dataset_loc = "../balanced_datasets"

transform = transforms.Compose([
    transforms.ToTensor()])

modified_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees = 90),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip()])

orig_dataset = torchvision.datasets.ImageFolder(root=dataset_loc, transform = transform)
modified_dataset = torchvision.datasets.ImageFolder(root=dataset_loc, transform = modified_transform)

classes = ["billete", "knife", "monedero", "pistol", "smartphone", "tarjeta"]
num_files = np.zeros([len(classes)])

# Count number of files for each class
# Also copy the original files
for img, label in orig_dataset:
    current_class = classes[label]
    path = new_dataset_loc + "/" + current_class + "/" + current_class + str(int(num_files[label])) + ".jpg"
    save_image(img, path)
    num_files[label] += 1

print("Finished counting number of files for each class: ")
print(num_files)

max_num_files = np.max(num_files)
max_index = np.argmax(num_files)
classes_balanced = 1

while (classes_balanced < len(classes)):
    for img, label in modified_dataset:
        if(num_files[label] < max_num_files):
            current_class = classes[label]
            path = new_dataset_loc + "/" + current_class + "/" + current_class + str(int(num_files[label])) + ".jpg"
            save_image(img, path)
            num_files[label] += 1
            if(num_files[label] == max_num_files - 1):
                classes_balanced += 1

print("Finished balancing classes")

balanced_dataset = torchvision.datasets.ImageFolder(root=new_dataset_loc, transform = transform)
balanced_num_files = np.zeros([len(classes)])

# Count number of files for each class
for img, label in balanced_dataset:
    balanced_num_files[label] += 1

print("Finished counting balanced number of files for each class: ")
print(balanced_num_files)

