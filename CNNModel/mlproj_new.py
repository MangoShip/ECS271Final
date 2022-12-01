import os
import time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import seaborn as sns
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchmetrics import Metric

dataset_loc = "../datasets/Sohas_weapon-Classification"

# https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend = "nccl", rank = rank, world_size = world_size)

def testtrain_acc(x, net, device):
  correct = 0
  total = 0
  with torch.no_grad():
      for data in x:
          images, labels = data
          images, labels = images.to(device), labels.to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

  return(100*correct/total)

class DataLoader(object):
    def __init__(self):
        super(DataLoader, self).__init__()

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(225),
            transforms.ColorJitter(brightness = 0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        orig_dataset = torchvision.datasets.ImageFolder(root=dataset_loc, transform=transform)
        tot_samples = len(orig_dataset)
        samples_test = int(0.2 * tot_samples)

        # https://discuss.pytorch.org/t/using-imagefolder-random-split-with-multiple-transforms/79899
        self.train, self.test = torch.utils.data.random_split(orig_dataset, [tot_samples - samples_test, samples_test])
        self.trainLoader = torch.utils.data.DataLoader(self.train, batch_size=32, num_workers=2, shuffle=False, sampler=DistributedSampler(self.train))
        self.testLoader = torch.utils.data.DataLoader(self.test, batch_size=32, num_workers=2, shuffle=False, sampler=DistributedSampler(self.test))

        self.train_acc = []
        self.test_acc = []
        self.loss_vals = []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, stride=2)
        self.pool1 = nn.MaxPool2d(4, stride=2)
        self.conv2 = nn.Conv2d(96, 128, 3, stride=2, padding=2)
        self.pool2 = nn.MaxPool2d(4, stride=2)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 2056)
        self.fc2 = nn.Linear(2056,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,64)
        self.fc5 = nn.Linear(64,6)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class MetricLoss(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0, dtype=torch.float64))
        self.add_state("train_accuracy", default=torch.tensor(0, dtype=torch.float64))
        self.add_state("test_accuracy", default=torch.tensor(0, dtype=torch.float64))

    def update(self, loss, train_accuracy, test_accuracy):
        self.loss += torch.tensor(loss, dtype=torch.float64)
        self.train_accuracy += torch.tensor(train_accuracy, dtype=torch.float64)
        self.test_accuracy += torch.tensor(test_accuracy, dtype=torch.float64)

    def compute(self):
        return (torch.mean(self.loss)).item(), (torch.mean(self.train_accuracy)).item(), (torch.mean(self.test_accuracy)).item()

def main(rank, world_size):
    
    ddp_setup(rank, world_size)
    gpu_id = rank

    metric = MetricLoss()
    
    net = Net()
    net.metric = metric
    net.to(gpu_id)
    net = DDP(net, device_ids=[gpu_id])

    dataLoader = DataLoader()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    epoch_num = 100

    for epoch in range(epoch_num):  # loop over the dataset multiple times

        print("Epoch #%d:" % epoch)
        dataLoader.trainLoader.sampler.set_epoch(epoch)
        dataLoader.testLoader.sampler.set_epoch(epoch)
        
        # Start timer
        epoch_timer_start = time.perf_counter()
        train_timer_start = time.perf_counter()

        running_loss = 0.0
        for i, data in enumerate(dataLoader.trainLoader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(gpu_id), labels.to(gpu_id)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            '''if i % 249 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                # running_loss = 0.0'''

        # End timer
        train_timer_end = time.perf_counter()
        
        train_acc_timer_start = time.perf_counter()
        train_acc = testtrain_acc(dataLoader.trainLoader, net, gpu_id)
        train_acc_timer_end = time.perf_counter()

        test_acc_timer_start = time.perf_counter()
        test_acc = testtrain_acc(dataLoader.testLoader, net, gpu_id)
        test_acc_timer_end = time.perf_counter()

        metric(running_loss / len(dataLoader.trainLoader), train_acc, test_acc)
        total_loss, total_train_acc, total_test_acc = metric.compute()
        
        '''print("Individual Loss:", running_loss / len(dataLoader.trainLoader))
        print("Individual Train Acc:", train_acc)
        print("Individual Test Acc:", test_acc)
        print("Metric Loss: ", total_loss)
        print("Metric Train Acc:", total_train_acc)
        print("Metric Test Acc:", total_test_acc)'''
        metric.reset()

        dataLoader.loss_vals.append(total_loss)
        dataLoader.train_acc.append(total_train_acc)
        dataLoader.test_acc.append(total_test_acc)

        # End timer
        epoch_timer_end = time.perf_counter()
        '''print("Training Duration: %.3fs" % (train_timer_end - train_timer_start))
        print("Training Accuracy Duration: %.3fs" % (train_acc_timer_end - train_acc_timer_start))
        print("Testing Accuracy Duration: %.3fs" % (test_acc_timer_end - test_acc_timer_start))
        print("Epoch Duration: %.3fs" % (epoch_timer_end - epoch_timer_start))'''


    epochs = [i+1 for i in range(epoch_num)]
    plt.plot(epochs, dataLoader.loss_vals, marker="o", label="Total Loss")
    plt.legend()
    plt.xlabel("Number of Epoch")
    plt.savefig("total_loss.png")
    plt.clf()
    plt.plot(epochs, dataLoader.test_acc, marker="o", label="Test Accuracy")
    plt.plot(epochs, dataLoader.train_acc, marker="o", label="Train Accuracy")
    plt.legend()
    plt.xlabel("Number of Epoch")
    plt.savefig("accuracy.png")
    destroy_process_group()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print("Using", world_size, "GPUs")
    mp.spawn(main, args=(world_size,), nprocs = world_size)
    #main()