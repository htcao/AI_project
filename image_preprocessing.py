#import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np
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
    input = Variable(preprocessed_img, requires_grad = True)
    return input, img


def load_data(filename):
    data = np.load(filename)
    img = data[data.shape[0]//2]
    data = torch.from_numpy(data.astype(float)).float()
    data.unsqueeze_(0)
    input = Variable(data, requires_grad = True)
    return input, img
