import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2, json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import keras
import os
from keras import layers
import random
from keras.models import Sequential, model_from_json
from keras.layers import Dense,Dropout, Flatten,Conv2D,MaxPool2D


DestiPath="/home/royce/Desktop/Facultate/ShipDetect/mydts"


def Load_mode():
    model=keras.models.load_model('/home/royce/Desktop/Facultate/ShipDetect/SaveModel/ShipDecModel')
    return model

def Load_img():
    images,labels=[],[]
    count=0
    for file in os.listdir(DestiPath):
        if(count>5600):
            imgPath=os.path.join(DestiPath,file)
            img=cv2.imread(imgPath)
            label=imgPath[len(DestiPath)+1]
            images.append(img)
            labels.append(label)
        count+=1
        
    images=np.array(images,dtype=np.float32)/255.0
    labels=np.array(labels,dtype=np.float32)
    return images,labels
'''
def Load_img():
    images,labels=[],[]
    for file in os.listdir(DestiPath):
        if(count>5600):
            imgPath=os.path.join(DestiPath,file)
            img=cv2.imread(imgPath)
            label=imgPath[len(DestiPath)+1]
            images.append(img)
            labels.append(label)
        count+=1
        
    images=np.array(images,dtype=np.float32)/255.0
    labels=np.array(labels,dtype=np.float32)
    return images,labels
'''
def Load_img_predict():
    images,labels=[],[]
    count=0
    no_true=0
    no_false=0
    for file in os.listdir('/home/royce/Desktop/Facultate/ShipDetect/dataset/shipsnet/shipsnet'):
        if(count>2000):
            imgPath=os.path.join('/home/royce/Desktop/Facultate/ShipDetect/dataset/shipsnet/shipsnet',file)
            label=imgPath[len('/home/royce/Desktop/Facultate/ShipDetect/dataset/shipsnet/shipsnet')+1]
            if (no_true<200 and label=="1" ):
                img=cv2.imread(imgPath)
                images.append(img)
                labels.append(label)
                no_true+=1
            if(no_false<200 and label=="0"):
                img=cv2.imread(imgPath)
                images.append(img)
                labels.append(label)
                no_false+=1
        count+=1
        
    images=np.array(images,dtype=np.float32)/255.0
    labels=np.array(labels,dtype=np.float32)
    return images,labels

def check_some(images,labels,test_pred,number):
    lambda_func=lambda x: "ship" if x==1 else "no-ship"
    test_pred=np.argmax(test_pred,axis=1)
    for x in range(number):
        num=random.randint(0,399)
        plt.imshow(images[num])
        plt.title("This is: "+lambda_func(labels[num])+"\n Predicted as: "+lambda_func(test_pred[num]))
        plt.show()



if __name__=="__main__":

    model=Load_mode()
    images,labels=Load_img()
    print(images.shape,labels.shape)
    model.summary()
    
    labels_c=to_categorical(labels)
    tst_p=model.evaluate(images,labels_c)
    print("Accuraty:%f" %tst_p[1])

    images_p,labels_p=Load_img_predict()
    print(images_p.shape)
    test_pred=model.predict(images)
    check_some(images,labels,test_pred,20)

    