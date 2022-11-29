import argparse
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

from utils import *
from dataloader import DataLoader
from net import Net

def train(device, trainloader, net, criterion, optimizer):
    running_loss = 0.0
    training_size = 0
    training_total = 0
    training_correct = 0

    # Start timer
    timer_start = time.perf_counter()
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and labels
        inputs, labels = data

        # Transfer inputs and labels to device
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Get loss for current mini-batch
        running_loss += loss.item()
        training_size += 1

        # Get accuracy for current mini-batch
        _, predicted = torch.max(outputs.data, 1)
        training_total += labels.size(0)
        training_correct += (predicted == labels).sum().item()

    # End timer
    timer_end = time.perf_counter()

    # Print total loss, training accuracy, and training duration
    print("Total Loss: %f" % (running_loss / training_size))
    print("Training Accuracy: %d %%" % (100 * training_correct / training_total))
    print("Training Duration: %.3fs" % (timer_end - timer_start))

def test(device, testloader, net):
    testing_total = 0
    testing_correct = 0

    # Start timer
    timer_start = time.perf_counter()
    with torch.no_grad():
        for data in testloader:
            # Get the images and labels
            images, labels = data

            # Transfer images and labels to device
            images, labels = images.to(device), labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            testing_total += labels.size(0)
            testing_correct += (predicted == labels).sum().item()

    # End timer
    timer_end = time.perf_counter()

    # Print testing accuracy and testing duration
    print("Testing Accuracy: %d %%" % (100 * testing_correct / testing_total))
    print("Testing Duration: %.3fs" % (timer_end - timer_start))


def main():
    # Argument Parser
    parser = argparse.ArgumentParser(description='ECS271 Final Project')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='enables CUDA for GPU Acceleration')

    args = parser.parse_args()
    
    # Check if cuda will be used
    cuda_enabled = args.cuda and torch.cuda.is_available()

    if cuda_enabled:
        print("CUDA has been enabled")
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    num_epochs = 1
    dataLoader = DataLoader()
    net = Net().to(device)
    #displayImageDimensions('./datasets/Sohas_weapon-Classification');

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train and test model for every number of epoch
    for epoch in range(num_epochs):
        print("Epoch #%d:" % epoch)

        train(device, dataLoader.trainloader, net, criterion, optimizer)
        test(device, dataLoader.testloader, net)

if __name__ == '__main__':
    main()
