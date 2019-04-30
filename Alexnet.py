import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch.utils.model_zoo as model_zoo
import numpy as np

import copy
from collections import OrderedDict
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import os


class Alexnet(nn.Module):
    def __init__(self):
        super(Alexnet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    since = time.time()
    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0
    for epoch in range(10):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, 10 - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # Set model to training mode
            else:
                net.eval()  # Set model to evaluate mode
            running_loss = 0.0
            running_correct = 0
            # Iterate over data
            for features, labels in trainloader[phase]:
                features = features.to(device)
                labels = labels.view(-1, 1).to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(features)
                    # loss = criterion(outputs, labels)
                    loss = torch.mean(torch.clamp(1 - outputs*labels, min=0))
                    loss += 0.01 * torch.mean(net.classifier._modules['6'].weight ** 2)  # l2 penalty
                    predicted = 2*(outputs.data > 0).float()-1
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # print statistics
                running_loss += loss.item() * features.size(0)
                running_correct += torch.sum(predicted == labels)

            epoch_loss = running_loss / len(trainloader[phase].dataset)
            epoch_acc = running_correct.double() / len(trainloader[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(net.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    net.load_state_dict(best_model_wts)


def test(testloader, net, device):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.view(-1, 1).to(device)
            outputs = net(images)
            predicted = 2*(outputs.data > 0).float()-1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %d test images: %d %%' % (total, 
        100 * correct / total))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    pretrained = False
    save_flag = True
    data_control = np.load('Control_data_new.npy')
    arrays = [data_control[i] for i in range(data_control.shape[0])]
    data_control = np.concatenate(arrays, axis=0)
    data_control = data_control[:, np.newaxis, :, :]
    np.random.shuffle(data_control)
#    data_control = data_control[:, 4:7, :, :]
    data_control = torch.from_numpy(data_control.astype(float)).float()
    # y_control = torch.zeros(data_control.size(0), dtype=torch.float)
    y_control = -torch.ones(data_control.size(0), dtype=torch.float)
    data_control_train = data_control[0:int(data_control.size(0)*0.6)]
    data_control_val = data_control[int(data_control.size(0)*0.6):int(data_control.size(0)*0.8)]
    data_control_test =  data_control[int(data_control.size(0)*0.8):]
    y_control_train = y_control[0:int(data_control.size(0)*0.6)]
    y_control_val = y_control[int(data_control.size(0)*0.6):int(data_control.size(0)*0.8)]
    y_control_test = y_control[int(data_control.size(0)*0.8):]
    data_pd = np.load('PD_data.npy')
    arrays = [data_pd[i] for i in range(data_pd.shape[0])]
    data_pd = np.concatenate(arrays, axis=0)
    data_pd = data_pd[:, np.newaxis, :, :]
    np.random.shuffle(data_pd)
    # data_pd = data_pd[201:, :, :, :]
#    data_pd = data_pd[:, 4:7, :, :]
    data_pd = torch.from_numpy(data_pd.astype(float)).float()
    y_pd = torch.ones(data_pd.size(0), dtype=torch.float)
    data_pd_train = data_pd[0:int(data_pd.size(0)*0.6)]
    data_pd_val = data_pd[int(data_pd.size(0)*0.6):int(data_pd.size(0)*0.8)]
    data_pd_test =  data_pd[int(data_pd.size(0)*0.8):]
    y_pd_train = y_pd[0:int(data_pd.size(0)*0.6)]
    y_pd_val = y_pd[int(data_pd.size(0)*0.6):int(data_pd.size(0)*0.8)]
    y_pd_test = y_pd[int(data_pd.size(0)*0.8):]
    data_train = torch.cat((data_control_train, data_pd_train), 0)
    data_val = torch.cat((data_control_val, data_pd_val), 0)
    data_test = torch.cat((data_control_test, data_pd_test), 0)
    y_train = torch.cat((y_control_train, y_pd_train), 0)
    y_val = torch.cat((y_control_val, y_pd_val), 0)
    y_test = torch.cat((y_control_test, y_pd_test), 0)
    
    trainset = data_utils.TensorDataset(data_train, y_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=15,
                                              shuffle=True)
    
    valset = data_utils.TensorDataset(data_val, y_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=15, shuffle=False)
    
    Dataloader = {'train': trainloader, 'val': valloader}

    testset = data_utils.TensorDataset(data_test, y_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=15, shuffle=False)
                   
    net = Alexnet().to(device)
    if pretrained:
        net.load_state_dict(torch.load('./weights/alexnet_weight.pt'))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    train(Dataloader, net, criterion, optimizer, device)
    test(testloader, net, device)
    if not os.path.exists('./weights'):
        os.makedirs('weights')
    if save_flag:
        torch.save(net.state_dict(), './weights/alexnet_weight_svm.pt')
#    conv5_weights = net.state_dict()['features.12.weight'].cpu()
# #    image_weight = conv5_weights[0,:,:,:].numpy()
# #    image_weight = np.maximum(image_weight, 0)
# #    image_weight = np.mean(image_weight, axis=0)
# #    image_weight = (image_weight-np.min(image_weight))/(np.max(image_weight)-np.min(image_weight))
# #    plt.imshow(image_weight)

if __name__ == "__main__":
    main()
