import os
import time
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from utils import displayImageDimensions
from dataloader import DataLoader
from net import Net

def train(trainloader, net, criterion, optimizer):
    running_loss = 0.0
    training_size = 0
    training_total = 0
    training_correct = 0

    # Start timer
    timer_start = time.perf_counter()
    for i, data in enumerate(trainloader, 0):
        print(i)
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # get loss for current mini-batch
        running_loss += loss.item()
        training_size += 1

        # get accuracy for current mini-batch
        _, predicted = torch.max(outputs.data, 1)
        training_total += labels.size(0)
        training_correct += (predicted == labels).sum().item()

    # End timer
    timer_end = time.perf_counter()

    # Print total loss, training accuracy, and training duration
    print("Total Loss: %d" % (running_loss / training_size))
    print("Training Accuracy: %d %%" % (100 * training_correct / training_total))
    print("Training Duration: %fs" % (timer_end - timer_start))

def test(testloader, net):
    testing_total = 0
    testing_correct = 0

    # Start timer
    timer_start = time.perf_counter()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            testing_total += labels.size(0)
            testing_correct += (predicted == labels).sum().item()

    # End timer
    timer_end = time.perf_counter()

    # Print testing accuracy and testing duration
    print("Testing Accuracy: %d %%" % (100 * testing_correct / testing_total))
    print("Testing Duration: %fs" % (timer_end - timer_start))


def main():
    
    num_epochs = 1
    dataLoader = DataLoader()
    net = Net()
    #displayImageDimensions('./datasets/Sohas_weapon-Classification');

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train and test model for every number of epoch
    for epoch in range(num_epochs):
        print("Epoch #%d:" % epoch)

        train(dataLoader.trainloader, net, criterion, optimizer)
        test(dataLoader.testloader, net)

if __name__ == '__main__':
    main()
