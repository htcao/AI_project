import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch.utils.model_zoo as model_zoo
import numpy as np

import math
from collections import OrderedDict
import torch.utils.data as data_utils


class Alexnet(nn.Module):
    # Since the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3, 
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self):
        super(Alexnet, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 96, 11, stride=4),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            # Stage 2
            nn.Conv2d(96, 256, 5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            # Stage 3
            nn.Conv2d(256, 384, 3, stride=1, padding=1),
            nn.ReLU(),
            # Stage 4
            nn.Conv2d(384, 384, 3, stride=1, padding=1),
            nn.ReLU(),
            # Stage 5
            nn.Conv2d(384, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2)
        )
        self.fc = nn.Sequential(
            # fully connected layers
            # parameters still needs to be changed
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(10):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.view(-1, 1).to(device)
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
            labels = labels.view(-1, 1).to(device)
            outputs = net(images)
            predicted = (outputs.data > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#    device = torch.device('cpu')
    model_urls = {
        'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    }
    pretrained = False
    data_control = np.load('data.npy')
    data_control = torch.from_numpy(data_control.astype(float)).float()
    data_control = data_control[:, 4:7, :, :]
    y_control = torch.zeros(data_control.size(0), dtype=torch.float)
    data_control_train = data_control[0:int(data_control.size(0)*0.8)]
    data_control_test =  data_control[int(data_control.size(0)*0.8):]
    y_control_train = y_control[0:int(data_control.size(0)*0.8)]
    y_control_test = y_control[int(data_control.size(0)*0.8):]
    data_pd = np.load('PD_data.npy')
    data_pd = torch.from_numpy(data_pd.astype(float)).float()
    data_pd = data_pd[:, 4:7, :, :]
    y_pd = torch.ones(data_pd.size(0), dtype=torch.float)
    data_pd_train = data_pd[0:int(data_pd.size(0)*0.8)]
    data_pd_test =  data_pd[int(data_pd.size(0)*0.8):]
    y_pd_train = y_pd[0:int(data_pd.size(0)*0.8)]
    y_pd_test = y_pd[int(data_pd.size(0)*0.8):]
    data_train = torch.cat((data_control_train, data_pd_train), 0)
    y_train = torch.cat((y_control_train, y_pd_train), 0)
    data_test = torch.cat((data_control_test, data_pd_test), 0)
    y_test = torch.cat((y_control_test, y_pd_test), 0)
    
    trainset = data_utils.TensorDataset(data_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                              shuffle=True)

    testset = data_utils.TensorDataset(data_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False)
                   
    net = Alexnet().to(device)
    if pretrained:
        net.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)
    

if __name__ == "__main__":
    main()
