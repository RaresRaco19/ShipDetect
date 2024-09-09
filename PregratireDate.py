import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2, json
from tqdm import tqdm
import imgaug.augmenters as iaa
import imgaug.imgaug
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import keras
import os
import time
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential, model_from_json
from keras.layers import Dense,Dropout, Flatten,Conv2D,MaxPool2D
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

dataset="/home/royce/Desktop/Facultate/ShipDetect/dataset/shipsnet/shipsnet"

class_name=["no-ship","ship"]
class_name_labels={class_name:i for i,class_name in enumerate(class_name)}
SavePath="/home/royce/Desktop/Facultate/ShipDetect/mydts"

def load_data():
    images, labels =[],[]
    for file in tqdm(os.listdir(dataset)):
        img_path=os.path.join(dataset,file)
        img=cv2.imread(img_path)
        label=img_path[len(dataset)+1]
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)
        labels.append(label)
    images=np.array(images,dtype=np.float32)/255.0
    labels=np.array(labels,dtype=np.float32)
    return (images,labels)


def save_data(images,labels):
    imags=np.array(images,dtype=np.float32)*255.0
    for id,image in enumerate(imags):
        extension=str(labels[id])+"_"+str(id)+".png"
        img_path=os.path.join(SavePath,extension)
        cv2.imwrite(img_path,image)
    return True


def augment_add(images,seg: iaa.Sequential,labels):
    augmeted_images, augmente_labels =[],[]
    for idx,img in tqdm(enumerate(images)):
        if labels[idx]==1.0:
            image_aug_1=seg.augment_image(image=img)
            image_aug_2=seg.augment_image(image=img)
            augmeted_images.append(image_aug_1)
            augmeted_images.append(image_aug_2)
            augmente_labels.append(labels[idx])
            augmente_labels.append(labels[idx])
    augmeted_images=np.array(augmeted_images,dtype=np.float32)
    augmente_labels=np.array(augmente_labels,dtype=np.float32)
    return(augmeted_images,augmente_labels)

def create_model():
    model=Sequential()
    model.add(Conv2D(filters=32,kernel_size=(8,8),padding='Valid',activation='relu'))
    model.add(MaxPool2D(pool_size=(6,6),strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(filters=64,kernel_size=(4,4),padding='Same',activation='relu'))
    model.add(MaxPool2D(pool_size=(5,5),strides=(3,3)))

    model.add(Flatten())
    model.add(Dense(200,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(150,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation='sigmoid'))
    return model


def first_view(images,labels):
    _, Count=np.unique(labels,return_counts=True)
    df=pd.DataFrame(data=Count,index=['no-ship','ship'],columns=['Count'])
    print(df)
    df.plot.bar(rot=0)
    plt.title("Distributia claselor(original)")
    plt.savefig("/home/royce/Desktop/Facultate/ShipDetect/Figures/Dist(original).png")
    plt.close()
    plt.imshow(images[1509])
    plt.title(labels[1509])
    plt.savefig("/home/royce/Desktop/Facultate/ShipDetect/Figures/RandomImage.png")
    plt.close()

def verify_augdata(augment_images,augment_labels):
    print((augment_images.shape,augment_labels.shape))
    plt.subplot(1,3,1)
    plt.title(augment_labels[1569])
    plt.imshow(augment_images[1569])
    plt.subplot(1,3,2)
    plt.title(augment_labels[1570])
    plt.imshow(augment_images[1570])
    plt.subplot(1,3,3)
    plt.title(augment_labels[1571])
    plt.imshow(augment_images[1571])
    plt.savefig("/home/royce/Desktop/Facultate/ShipDetect/Figures/RandomAugImage.png")
    plt.close()
    
def conca_view(images,labels):
    print((images.shape,labels.shape))
    print(labels[5000])
    _, count=np.unique(labels,return_counts=True)
    dfa=pd.DataFrame(data=count,index=['no-ship','ship'],columns=['Count'])
    dfa.plot.bar(rot=0)
    plt.title("Distributia claselor dupa augmentare")
    plt.savefig("/home/royce/Desktop/Facultate/ShipDetect/Figures/Dist(augment).png")
    plt.close()

def suffle_set(images,labels):
    np.random.seed(42)
    np.random.shuffle(images)
    np.random.seed(42)
    np.random.shuffle(labels)
    return images,labels

def check_suffle(iamges,labels):
    plt.subplot(1,3,1)
    plt.imshow(images[657])
    plt.title(labels[657])
    plt.subplot(1,3,2)
    plt.imshow(images[4000])
    plt.title(labels[4000])
    plt.subplot(1,3,3)
    plt.imshow(images[2653])
    plt.title(labels[2653])
    plt.savefig("/home/royce/Desktop/Facultate/ShipDetect/Figures/RandomImageFromAll.png")
    plt.close()

def divide_set(images,labels):
    size=len(images)
    train=int(0.7*size)
    validation=int(0.2*size)
    test=int(0.1*size)
    labels=to_categorical(labels)
    train_images,train_labels=images[:train],labels[:train]
    validation_images,validation_labels=images[train:(validation+train)],labels[train:(train+validation)]
    test_images,test_labels=images[-test:],labels[-test:]
    return train_images,train_labels,validation_images,validation_labels,test_images,test_labels

def distrib_train(train_labels):
    _,con=np.unique(train_labels,return_counts=True)
    df_train=pd.DataFrame(data=con,index=["ship","no-ship"],columns=["Count"])
    df_train.plot.bar(rot=0)
    plt.savefig("/home/royce/Desktop/Facultate/ShipDetect/Figures/Dist(trainset)")
    plt.close()

def train(model,eps):
    opt=Adam(learning_rate=0.002,beta_1=0.9,beta_2=0.999)
    model.compile(optimizer=opt,loss="binary_crossentropy",metrics=["accuracy"])
    history=model.fit(train_images,train_labels,epochs=eps,validation_data=(validation_images,validation_labels))
    return history


if __name__=='__main__':
    (images,labels)=load_data()
    lambda_func=lambda x: print("ship") if x==1 else print("no-ship")
    print((images.shape,labels.shape))
    
    first_view(images,labels)
    Aug= True
    seq=iaa.Sequential([iaa.Fliplr(0.15,5),iaa.LinearContrast((0.75,1.5)),iaa.Multiply((0.8,1.2),per_channel=0.2),iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, rotate=(-25, 25), shear=(-8, 8) )],random_order=True)
    (augment_images,augment_labels)=augment_add(images,seq,labels)
    
    verify_augdata(augment_images,augment_labels)
    images=np.concatenate([images,augment_images])
    labels=np.concatenate([labels,augment_labels])
   
   
    conca_view(images,labels)

    if(save_data(images,labels)):
        print("Save successfully")
    else:
        print("Save doesnt work")
    
    images,labels=suffle_set(images,labels)
    
    
    check_suffle(images,labels)

    
    train_images,train_labels,validation_images,validation_labels,test_images,test_labels=divide_set(images,labels)

    
    distrib_train(train_labels)
    
    model=create_model()
    
    history=train(model,50)
    model.save('/home/royce/Desktop/Facultate/ShipDetect/SaveModel/ShipDecModel')


   

    


   
