import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.utils.data as data_utils

import math
from collections import OrderedDict


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 5 == 4:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data = np.load('data.npy')
    data = data[:, 0:3, :, :]
    data = torch.from_numpy(data.astype(float)).float()
    y = torch.zeros(data.size(0), dtype=torch.long)

    trainset = data_utils.TensorDataset(data, y)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True)

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                    download=True, transform=transform)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=100,
    #                                      shuffle=False)
    net = torchvision.models.alexnet(pretrained=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    # test(testloader, net, device)
    

if __name__ == "__main__":
    main()
