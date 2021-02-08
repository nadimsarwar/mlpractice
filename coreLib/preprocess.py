# -*- coding: utf-8 -*-
'''
    @author: Nadim Sarwar
'''
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
def get_Xy(_path,
            lower_thresh,
            nb_classes,
            dim=64):
    '''
        preprocesses an image and label
        args:
            _path           = the path of test dir / train dir where all the images are separated with corresponding labels
            lower_thresh    = the threshold value for determining the label    
            nb_classes      = the total number of classes
            dim             = the size of images to resize to
        returns:
            X   =   stacked images with given dimention
            y   =   stacked one-hot encoded labels
        preprocess ops:
            * data shuffle
        image ops:
            * read
            * resize
            * thresh
            * image to tensor
            * convert to float (special case)
        label ops:
            * extract label from path
            * convert to float (special case)
                
    '''
    X=[]
    y=[]
    img_paths=glob(os.path.join(_path,'*/*.bmp'))
    # shuffle
    random.shuffle(img_paths)
    # iterate
    for img_path in tqdm(img_paths):
        # extract int label from full path
        label=int(os.path.basename(os.path.dirname(img_path)))-lower_thresh
        # encode label
        encoded_label=np.zeros(nb_classes,dtype=np.float)
        encoded_label[label]=1
        # update
        y.append(encoded_label)

        # image ops
        # read
        img=cv2.imread(img_path,0)
        # resize
        img=cv2.resize(img,(dim,dim))
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img,(5,5),0)
        _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # image to tensor
        x=np.expand_dims(img,axis=-1)
        x=np.expand_dims(x,axis=0)
        # float conversion
        x=x/255.0
        x=1-x
        # debug
        #plt.imshow(np.squeeze(x))
        #plt.show()
        # update
        X.append(x)
        
    # stacking
    X=np.vstack(X)
    y=np.vstack(y)
    return X,y
#---------------------------------------------------------------
