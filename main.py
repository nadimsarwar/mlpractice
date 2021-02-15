#!/home/nadim/Desktop/ml/mlpractice/venv/bin/python3
# -*-coding: utf-8 -
'''
    @author: Nadim Sarwar
'''
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
import random
import argparse

from glob import glob
from tqdm import tqdm

from coreLib.utils import LOG_INFO,create_dir
from coreLib.data import Processor
#---------------------------------------------------------------
#---------------------------------------------------------------

def main(args):
    '''
        preprocesses data for training
        args:
            data_path   =   the location of folder that contains test and train 
            save_path   =   path to save the tfrecords
            fmt         =   format of the image
            data_dim    =   dimension to resize the images
            image_type  =   type of image (grayscale,rgb,binary)
            data_size   =   the size of tfrecords
            label_den   =   label denoter (by default : train)
    '''
    data_path   =   args.data_path
    save_path   =   args.save_path
    fmt         =   args.fmt
    data_dim    =   int(args.data_dim)
    image_type  =   args.image_type
    data_size   =   int(args.data_size)
    label_den   =   args.label_den
    processor_obj=Processor(data_path,save_path,fmt,data_dim,image_type,data_size,label_den)
    processor_obj.process()

if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("preprocessing script image classification datasets")
    parser.add_argument("--data_path", help="Path of the data folder that contains Test and Train")
    parser.add_argument("--save_path", help="Path to save the tfrecords")
    parser.add_argument("--fmt",help =   "format of the image")
    parser.add_argument("--data_dim",help ="dimension to resize the images")
    parser.add_argument("--image_type",help ="type of image (grayscale,rgb,binary)")
    parser.add_argument("--data_size",required=False,default=1024,help ="the size of tfrecords")
    parser.add_argument("--label_den",required=False,default='train',help ="label denoter (by default : train)")
    args = parser.parse_args()
    main(args)
    