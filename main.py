from cnn import cnn

import numpy as np
import cv2

import sys
import os
from glob import glob

from tqdm import tqdm

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import keras 
from keras.utils import to_categorical

def read_dataset():
    for myfile in glob("cropped\\cat_????.png"):
       print(myfile) 

def read_file(img_file):
    img = cv2.imread(img_file,cv2.IMREAD_GRAYSCALE)
    ret,img_bin = cv2.threshold(img,170,255,cv2.THRESH_BINARY_INV)
    # img_bin = img_bin/255
    return img_bin

if __name__ == "__main__":
    if not(os.path.exists("bin")):
        os.mkdir("bin")

    x_all = []
    y_all = []        
    for myimgfile in tqdm(glob("data\\cat*.png")):
        myimg = read_file(myimgfile)
        myimg =cv2.resize(myimg,(28,28))
        x_all.append(myimg)
        y_all.append(0)

    for myimgfile in tqdm(glob("data\\rabbit*.png")):
        myimg = read_file(myimgfile)
        myimg =cv2.resize(myimg,(28,28))

        x_all.append(myimg)
        y_all.append(1)

    x_all = np.array(x_all).reshape(len(x_all),28,28,1)/255
    y_all = np.array(y_all)

    y_all = to_categorical(y_all,2)

    print(x_all.shape)
    print(y_all.shape)

    (x_train, x_test, y_train, y_test) =\
        train_test_split(x_all, y_all, 
        test_size=0.3, random_state=0)
    
    mymodel = cnn()
    mymodel.fit(x_train,y_train,
    batch_size=128,epochs=5,verbose=1,)

    mymodel.save("mymodel.h5")

    score = mymodel.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])