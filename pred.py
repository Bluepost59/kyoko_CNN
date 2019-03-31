import numpy as np 
import cv2

import os 
import sys 
from glob import glob 
from tqdm import tqdm

import keras 
from keras.models import load_model

if __name__=="__main__":
    mymodel = load_model("mymodel.h5")

    myimg = cv2.imread("myfile.png",0)
    myimg = cv2.resize(myimg,(28,28))
    myimg = myimg/255

    myimg = myimg.reshape((1,28,28,1))

    print(mymodel.predict(myimg))



