import os
import shutil
import pydicom
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2 as cv

def Normalization(x):
    Min=np.min(x)
    Max=np.max(x)
    x = (x - Min) / (Max-Min)
    return x

foldernames = os.listdir('./PD_60/')
res = []
resage = []
resweight=[]
ressex=[]
resecho=[]
for folder in foldernames:
    try:
        filenames = os.listdir('./PD_60/' + folder)
        images = [pydicom.dcmread(os.path.join('./PD_60/' + folder, filename)) for filename in filenames]
        # sort the files by InstanceNumber
        images.sort(key=lambda x: int(x.InstanceNumber))
        b=[]
        age=[]
        sex=[]
        weight=[]
        echo=[]
        for image in images:
            try:
                M,N=np.shape(image.pixel_array)
                L=min(M,N)
                im=image.pixel_array[int((M/2-L/2)):int((M/2+L/2)),int((N/2-L/2)):int((N/2+L/2))]
                im=cv.resize(im,(224,224))
                im=Normalization(im)
                b.append(im)
                # plot the gray scale image
                plt.imshow(im, cmap='gray')
                plt.axis('off')
                # save the images in 'results' folder and each image is named by its InstanceNumber
                #plt.savefig('./res/' + str(image.InstanceNumber-54) + '.png')
                age.append(int((image.PatientAge)[1:3]))
                weight.append(int(image.PatientWeight))
                if image.EchoTime=='101':
                    echo.append(1)
                else:
                    echo.append(0)
                if image.PatientSex=='M':
                    sex.append(0)
                else:
                    sex.append(1)
                    print(image.PatientSex)
            except:
                # if there is some error above, print the InstanceNumber
                print('Cant import ' + str(image.InstanceNumber))
        age=np.stack(age,axis=0)
        weight=np.stack(weight,axis=0)
        sex=np.stack(sex,axis=0)
        echo=np.stack(echo,axis=0)
        b=np.stack(b,axis=0)
        if np.shape(b[26:36]) == (10,224,224):
            res.append(b[26:36])
            resage.append(age[26:36])
            resweight.append(weight[26:36])
            ressex.append(sex[26:36])
            resecho.append(echo[26:36])
    except:
        print('Cant import ' + folder)
resage = np.stack(resage,axis=0)
resweight = np.stack(resweight,axis=0)
ressex = np.stack(ressex,axis=0)
resecho = np.stack(resecho,axis=0)
res = np.stack(res,axis=0)
np.save('./PD data/PD_data', res , allow_pickle=True, fix_imports=True)
np.save('./PD data/age', resage, allow_pickle=True, fix_imports=True)
np.save('./PD data/weight', resweight, allow_pickle=True, fix_imports=True)
np.save('./PD data/sex', ressex , allow_pickle=True, fix_imports=True)
np.save('./PD data/echotime', resecho , allow_pickle=True, fix_imports=True)