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
from coreLib.data import create_df
#---------------------------------------------------------------
#---------------------------------------------------------------

def main(args):
    '''
        preprocesses data for training
        args:
            data_path   =   the location of folder that contains test and train in CDB-3.1.2
            save_path   =   path to save the tfrecords
    '''
    # input
    data_path=args.data_path
    train_path =os.path.join(data_path,'Train')
    test_path  =os.path.join(data_path,'Test')
    # image paths
    LOG_INFO("Collecting Train Images")
    train_img_paths=[img_path for img_path in tqdm(glob(os.path.join(train_path,"*/*.bmp")))]
    random.shuffle(train_img_paths)

    LOG_INFO("Collecting Test Images")
    test_img_paths=[img_path for img_path in tqdm(glob(os.path.join(test_path,"*/*.bmp")))]
    random.shuffle(train_img_paths)
    
    # tfrecord directories 
    save_dir        =   create_dir(args.save_path,'tfrecords')
    train_save_dir  =   create_dir(save_dir,'train')
    test_save_dir   =   create_dir(save_dir,'test')
    # create tf records
    create_df(train_img_paths,train_save_dir)
    create_df(test_img_paths,test_save_dir)
    
if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("preprocessing script of CDB-3.1.2")
    parser.add_argument("data_path", help="Path of the data folder that contains Test and Train")
    parser.add_argument("save_path", help="Path to save the tfrecords")
    args = parser.parse_args()
    main(args)
    