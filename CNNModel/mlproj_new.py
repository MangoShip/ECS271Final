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

dataset_loc = ""

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
train, test = torch.utils.data.random_split(orig_dataset, [tot_samples - samples_test, samples_test])
trainLoader = torch.utils.data.DataLoader(train, batch_size=32, num_workers=2, shuffle=True)
testLoader = torch.utils.data.DataLoader(test, batch_size=32, num_workers=2)

train_acc = []
test_acc = []
loss_vals = []

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(100):  # loop over the dataset multiple times

        print("Epoch #%d:" % epoch)
        
        # Start timer
        epoch_timer_start = time.perf_counter()
        train_timer_start = time.perf_counter()

        running_loss = 0.0
        for i, data in enumerate(trainLoader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
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
        train_acc.append(testtrain_acc(trainLoader, net, device))
        train_acc_timer_end = time.perf_counter()

        test_acc_timer_start = time.perf_counter()
        test_acc.append(testtrain_acc(testLoader, net, device))
        test_acc_timer_end = time.perf_counter()

        loss_vals.append(running_loss / len(trainLoader))

        # End timer
        epoch_timer_end = time.perf_counter()
        print("Training Duration: %.3fs" % (train_timer_end - train_timer_start))
        print("Training Accuracy Duration: %.3fs" % (train_acc_timer_end - train_acc_timer_start))
        print("Testing Accuracy Duration: %.3fs" % (test_acc_timer_end - test_acc_timer_start))
        print("Epoch Duration: %.3fs" % (epoch_timer_end - epoch_timer_start))


    epochs = [i+1 for i in range(100)]
    plt.plot(epochs, test_acc, marker="o", label="Test Accuracy")
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, loss_vals, marker="o", label="Total Loss")
    plt.legend()
    plt.xLabel("Number of Epoch")
    plt.savefig("test.png")

if __name__ == '__main__':
    main()