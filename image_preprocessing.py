import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv

# filename = "PPMI_3101_MR_Axial_PD-T2_TSE_FS_br_raw_20141027100405111_74_S234691_I449541.dcm"
# folder = './PPMI/3101/Axial_PD-T2_TSE_FS/2014-08-20_10_57_57.0/S234691'
# ds = pydicom.dcmread(os.path.join(folder, filename))  # plan dataset
# print(ds)
# a = ds.pixel_array
# a = a/np.amax(a)*255
# # a = cv.resize(a, (224, 224), interpolation = cv.INTER_AREA)
# plt.imshow(a, cmap='gray')
# plt.axis('off')
# plt.show()


folder = './PPMI/3101/Axial_PD-T2_TSE_FS/2014-08-20_10_57_57.0/S234691'  # name of folder
filenames = os.listdir(folder)
images = [pydicom.dcmread(os.path.join(folder, filename)) for filename in filenames]
# sort the files by InstanceNumber
images.sort(key=lambda x: int(x.InstanceNumber))
for image in images:
    try:
        # plot the gray scale image
        plt.imshow(image.pixel_array, cmap='gray')
        plt.axis('off')
        # save the images in 'results' folder and each image is named by its InstanceNumber
        plt.savefig('./results/' + str(image.InstanceNumber) + '.png')
    except:
        # if there is some error above, print the InstanceNumber
        print('Cant import ' + str(image.InstanceNumber))
