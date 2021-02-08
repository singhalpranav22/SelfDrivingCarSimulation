import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from sklearn.utils import shuffle
import matplotlib.image as mpimg
import imgaug.augmenters as iaa
import cv2
import random

def getEnd(name):
    return name.split('/')[-1]  # return last index of the file name after splitting


def importData(path):
    colums = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=colums)
    data['Center'] = data['Center'].apply(getEnd)
    # print(data['Center'][0])
    print('Total Images imported:', data.shape[0])
    return data


def balanceData(data, display=True):
    nBins = 31
    samplesPerBin = 500
    hist, bins = np.histogram(data['Steering'], nBins)
    center = (bins[:-1] + bins[1:]) / 2
    if display:
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (800, 800))
        plt.show()
    removedIndexes = []
    for i in range(nBins):
        binList = []
        for j in range(len(data['Steering'])):
            if data['Steering'][j] >= bins[i] and data['Steering'][j] <= bins[i + 1]:
                binList.append(j)
        binList = shuffle(binList)
        binList = binList[1000:]
        removedIndexes.extend(binList)
    print('Removed Images = ', len(removedIndexes))
    data.drop(data.index[removedIndexes], inplace=True)
    if display:
        hist, _ = np.histogram(data['Steering'], nBins)
        plt.bar(center, hist, width=0.06)
        plt.plot((-1, 1), (1000, 1000))
        plt.show()
    return data
def loadData(path,data):
    imagesPath = []
    steerings = []
    for i in range(len(data)):
        indexdata=data.iloc[i]
        # print(indexdata)
        imagesPath.append(os.path.join(path,"IMG",indexdata[0]))
        steerings.append((float(indexdata[3])))
    imagesPath = np.asarray(imagesPath)
    steerings = np.asarray(steerings)
    return imagesPath,steerings
def augmentImage(imgPath,steering):
    img = mpimg.imread(imgPath)
    # PANning the image
    if np.random.rand()<0.5:
        pan = iaa.Affine(translate_percent={'x':(-0.1,0.1),'y':(-0.1,0.1)})
        img=pan.augment_image(img)
    # zooming image
    if np.random.rand()<0.5:
        zoom = iaa.Affine(scale=(1,1.2))
        img = zoom.augment_image(img)
    # changing the brightness
    if np.random.rand()<0.5:
        brightness = iaa.Multiply((0.4,1.2))
        img = brightness.augment_image(img)
    #flipping the image
    if np.random.rand()<0.5:
        img = cv2.flip(img,1)
        steering = -steering
    return img,steering
def preprocessing(img):
    img = img[60:135,:,:]
    # according to Nvidia's standards
    img = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img,(3,3),0)
    img = cv2.resize(img,(200,66))
    img = img/255
    return img
im = preprocessing(mpimg.imread("test.jpg"))
plt.imshow(im)
plt.show()
def batchGen(imgPath,steering,batchSize,trainFlag=1):
    while True:
        imgBatch = []
        steeringsBatch = []
        for i in range((batchSize)):
            index = random.randint(0,len(imgPath)-1)
            if trainFlag:
                img,steer=augmentImage(imgPath[index],steering[index])
            else:
                img=mpimg.imread(imgPath[index])
                steer=steering[index]
            img = preprocessing(img)
            imgBatch.append(img)
            steeringsBatch.append(steer)
        yield(np.asarray(imgBatch),np.asarray(steeringsBatch))

