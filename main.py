import os
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

from utils import displayImageDimensions
from dataloader import DataLoader
from net import Net

def train():
    # Insert code for training
    x = 0


def test():
    # Insert code for testing
    x = 0


def main():

    num_epochs = 1
    dataLoader = DataLoader()
    net = Net()
    #displayImageDimensions('./datasets/Sohas_weapon-Classification');

    for epoch in range(num_epochs):
        # Train the model for number of epochs
        x = 0

if __name__ == '__main__':
    main()
