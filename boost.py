import numpy as np 
import cv2
import os
import sys
from glob import glob

def boost():
    if not(os.path.exists("data")):
        os.mkdir("data")

    for myfile in glob("cropped\\*.png"):    
        origin = cv2.imread(myfile,cv2.IMREAD_GRAYSCALE)
        origin_size = origin.shape

        mean = 0

        for i in range(20):
            sigma = 15
            noize = np.random.normal(mean,sigma,origin_size)
            noize = noize.reshape(origin_size)

            res = origin + noize

            res_title = "data\\{0}_{1:04d}.png".format(
                myfile.split("\\")[-1].split(".")[0],i)
            print(res_title)
            cv2.imwrite(res_title,res)

            res_hflip = cv2.flip(res,1)
            res_title = "data\\{0}_{1:04d}_h.png".format(
                myfile.split("\\")[-1].split(".")[0],i)
            print(res_title)
            cv2.imwrite(res_title,res_hflip)

            res_vflip = cv2.flip(res,0)
            res_title = "data\\{0}_{1:04d}_v.png".format(
                myfile.split("\\")[-1].split(".")[0],i)
            print(res_title)
            cv2.imwrite(res_title,res_vflip)

            res_hvflip = cv2.flip(res_hflip,0)
            res_title = "data\\{0}_{1:04d}_hv.png".format(myfile.split("\\")[-1].split(".")[0],i)
            print(res_title)
            cv2.imwrite(res_title,res_hvflip)

            for i_size in range(5,15):
                avg_shape = (i_size,i_size)

                blur_img = cv2.blur(res,avg_shape)
                title = "data\\{0}_{1:04d}_blur{2:02d}.png".format(myfile.split("\\")[-1].split(".")[0],i,i_size)
                cv2.imwrite(title,res)

                blur_img = cv2.blur(res_hflip,avg_shape)
                title = "data\\{0}_{1:04d}_blur{2:02d}h.png".format(myfile.split("\\")[-1].split(".")[0],i,i_size)
                cv2.imwrite(title,res_hflip)

                blur_img = cv2.blur(res_hvflip,avg_shape)
                title = "data\\{0}_{1:04d}_blur{2:02d}hv.png".format(myfile.split("\\")[-1].split(".")[0],i,i_size)
                cv2.imwrite(title,res_hvflip)

                blur_img = cv2.blur(res_vflip,avg_shape)
                title = "data\\{0}_{1:04d}_blur{2:02d}v.png".format(myfile.split("\\")[-1].split(".")[0],i,i_size)
                cv2.imwrite(title,res_vflip)



if __name__ =="__main__":
    boost()