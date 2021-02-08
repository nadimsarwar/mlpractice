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
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from tqdm import tqdm
from .utils import LOG_INFO
#---------------------------------------------------------------
# fixed variables
#---------------------------------------------------------------
DATA_NUM=1024
IMG_DIM =128
IDEN_START=172
#---------------------------------------------------------------
# data functions
#---------------------------------------------------------------
# feature fuctions
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
      return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
# data specific functions
def get_label(img_path):
    '''
        get label from data path
        args:
            img_path    =   path of the image to get label
    '''
    _base  =  os.path.dirname(img_path)
    _label =  os.path.basename(_base)
    _label =  int(_label) - IDEN_START
    return _label    


def to_tfrecord(img_paths,
                mode_dir,
                r_num):
    '''
      Creates tfrecords from Provided Image Paths
      args:
        img_paths   =   specific number of image paths
        mode_dir    =   location to save the tfrecords
        r_num       =   record number
    '''
    tfrecord_name=f'{r_num}.tfrecord'
    tfrecord_path=os.path.join(mode_dir,tfrecord_name) 
    LOG_INFO(tfrecord_path)

    with tf.io.TFRecordWriter(tfrecord_path) as writer:    
        
        for img_path in tqdm(img_paths):
            
            #label
            label=get_label(img_path)
            # image ops
            # read
            img=cv2.imread(img_path,0)
            # resize
            img=cv2.resize(img,(IMG_DIM,IMG_DIM))
            # Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(img,(5,5),0)
            _,img = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # Png encoded data
            _,img_coded = cv2.imencode('.png',img)
            # Byte conversion
            image_png_bytes = img_coded.tobytes()

            # feature desc
            data ={ 'image':_bytes_feature(image_png_bytes),
                    'label':_int64_feature(label)
            }
            
            features=tf.train.Features(feature=data)
            example= tf.train.Example(features=features)
            serialized=example.SerializeToString()
            writer.write(serialized)  
            
def create_df(img_paths,save_dir):
    '''
        tf record wrapper
        args:
            img_paths   =   all image paths for a mode
            save_dir    =   location to save the tfrecords
    '''
    for idx in range(0,len(img_paths),DATA_NUM):
        _paths=img_paths[idx:idx+DATA_NUM]
        to_tfrecord(_paths,save_dir,idx//DATA_NUM)
        