#import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch.autograd import Variable
#import cv2 as cv

def img_preprocessing(folder_path):
    foldernames = os.listdir(folder_path)
    images = [pydicom.dcmread(folder_path+filename) for filename in filenames]
    images.sort(key=lambda x: int(x.InstanceNumber))  
    imgs = []
    for image in images:
        im=image.pixel_array[int((M/2-L/2)):int((M/2+L/2)),int((N/2-L/2)):int((N/2+L/2))]
        im=cv.resize(im,(224,224))
        imgs.append(im)
    imgs = np.stack(imgs, axis=0)
    imgs = torch.from_numpy(imgs)
    img = imgs[imgs.shape[0]//2, :, :]
    imgs.unsqueeze_(0)
    input = Variable(imgs, requires_grad = True)
    return input, img


def load_data(filename):
    data = np.load(filename)
    data = data[50, 4:7, :, :]
#    data = np.uint8(data/np.max(data)*255)
#    img = data[data.shape[0]//2]
    img = data
    data = torch.from_numpy(data.astype(float)).float()
    data.unsqueeze_(0)
    input = Variable(data, requires_grad = True)
    return input, img
