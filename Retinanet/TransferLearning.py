import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import random
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import time
import matplotlib.patches as patches

path_to_detection_dataset = "../datasets/Sohas_weapon-Detection/"

def generate_label(obj):

    if obj.find('name').text == "billete":

        return 0

    elif obj.find('name').text == "knife":

        return 1

    elif obj.find('name').text == "monedero":

        return 2

    elif obj.find('name').text == "pistol":

        return 3

    elif obj.find('name').text == "smartphone":

        return 4

    elif obj.find('name').text == "tarjeta":

        return 5

    return -1

def plot_image_from_output(img, annotation):
    
    img = img.cpu().permute(1,2,0)
    
    rects = []

    for idx in range(len(annotation["boxes"])):
        xmin, ymin, xmax, ymax = annotation["boxes"][idx]

        if annotation['labels'][idx] == 0 :
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')
        
        elif annotation['labels'][idx] == 1 :
            
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='g',facecolor='none')
            
        else :
        
            rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='orange',facecolor='none')

        rects.append(rect)

    return img, rects

def generate_target(file):

    tree = ET.parse(file)
    root = tree.getroot()
    objects = root.findall('object')

    boxes = []
    labels = []

    for obj in objects:
      bndbox = obj.find('bndbox')
      xmin = int(bndbox.find('xmin').text)
      ymin = int(bndbox.find('ymin').text)
      xmax = int(bndbox.find('xmax').text)
      ymax = int(bndbox.find('ymax').text)
      bbox = (xmin, ymin, xmax, ymax)
      
      boxes.append(bbox)
      labels.append(generate_label(obj))

      boxes = torch.as_tensor(boxes, dtype=torch.float32) 
      labels = torch.as_tensor(labels, dtype=torch.int64) 

      target = {}
      target["boxes"] = boxes
      target["labels"] = labels
      
      return target

def collate_fn(batch):
    return tuple(zip(*batch))

class MyDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        # self.path = path
        # self.imgs = list(sorted(os.listdir(self.path)))
        # self.transform = transform
        self.path = image_paths
        self.image_paths = list(sorted(os.listdir(self.path)))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        file_image = self.image_paths[idx]
        file_label = self.image_paths[idx].split(".")[-2] + '.xml'
        img_path = os.path.join(self.path, file_image)
        
        if 'test' in self.path:
            label_path = os.path.join(path_to_detection_dataset+"annotations_test/xmls/", file_label)
        else:
            label_path = os.path.join(path_to_detection_dataset+"annotations/xmls/", file_label)

        # img = Image.open(img_path).convert("RGB")
        img = Image.open(img_path).convert("RGB")
        target = generate_target(label_path)
        
        to_tensor = torchvision.transforms.ToTensor()

        if self.transform:
            img, transform_target = self.transform(np.array(img), np.array(target['boxes']))
            target['boxes'] = torch.as_tensor(transform_target)

        # change to tensor
        img = to_tensor(img)

        return img, target

random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]

transform = transforms.Compose([
                                transforms.CenterCrop(64),
                                transforms.RandomHorizontalFlip(p=0.5),
                                transforms.RandomApply(random_transforms, p=0.3),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dataset = MyDataset(path_to_detection_dataset+'images/')
test_dataset = MyDataset(path_to_detection_dataset+'images_test/')

data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

retina = torchvision.models.detection.retinanet_resnet50_fpn(num_classes = 6, pretrained=False, pretrained_backbone = True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_epochs = 5
retina.to(device)
    
# parameters
params = [p for p in retina.parameters() if p.requires_grad] # select parameters that require gradient calculation
optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9)

len_dataloader = len(data_loader)

# about 4 min per epoch on Colab GPU
for epoch in range(num_epochs):
    start = time.time()
    retina.train()
    print(epoch)

    i = 0    
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = retina(images, targets) 

        losses = sum(loss for loss in loss_dict.values()) 

        i += 1

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses 
    print(epoch_loss, f'time: {time.time() - start}')

torch.save(retina.state_dict(),f'retina_{num_epochs}.pt')
# Load the saved model
# retina.load_state_dict(torch.load(f'retina_{num_epochs}.pt'))
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# retina.to(device)

def make_prediction(model, img, threshold):
    model.eval()
    preds = model(img)
    for id in range(len(preds)) :
        idx_list = []

        for idx, score in enumerate(preds[id]['scores']) :
            if score > threshold : #select idx which meets the threshold
                idx_list.append(idx)

        preds[id]['boxes'] = preds[id]['boxes'][idx_list]
        preds[id]['labels'] = preds[id]['labels'][idx_list]
        preds[id]['scores'] = preds[id]['scores'][idx_list]


    return preds

from tqdm import tqdm

labels = []
preds_adj_all = []
annot_all = []

for im, annot in tqdm(test_data_loader, position = 0, leave = True):
    im = list(img.to(device) for img in im)
    #annot = [{k: v.to(device) for k, v in t.items()} for t in annot]

    for t in annot:
        labels += t['labels']

    with torch.no_grad():
        preds_adj = make_prediction(retina, im, 0.5)
        preds_adj = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in preds_adj]
        preds_adj_all.append(preds_adj)
        annot_all.append(annot)

nrows = 8
ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*4))

batch_i = 0
for im, annot in test_data_loader:
    pos = batch_i * 4 + 1
    for sample_i in range(len(im)) :
        
        img, rects = plot_image_from_output(im[sample_i], annot[sample_i])
        axes[(pos)//2, 1-((pos)%2)].imshow(img)
        for rect in rects:
            axes[(pos)//2, 1-((pos)%2)].add_patch(rect)
        
        img, rects = plot_image_from_output(im[sample_i], preds_adj_all[batch_i][sample_i])
        axes[(pos)//2, 1-((pos+1)%2)].imshow(img)
        for rect in rects:
            axes[(pos)//2, 1-((pos+1)%2)].add_patch(rect)

        pos += 2

    batch_i += 1
    if batch_i == 4:
        break

# remove xtick, ytick
for idx, ax in enumerate(axes.flat):
    ax.set_xticks([])
    ax.set_yticks([])

colnames = ['True', 'Pred']

for idx, ax in enumerate(axes[0]):
    ax.set_title(colnames[idx])

plt.tight_layout()
plt.show()