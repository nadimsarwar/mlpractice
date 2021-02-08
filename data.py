#!/home/nadim/Desktop/ml/mlpractice/venv/bin/python3
# -*-coding: utf-8 -
'''
    @author: Nadim Sarwar
'''
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
import os
import argparse
from coreLib.utils import LOG_INFO
from coreLib.preprocess import get_Xy
#---------------------------------------------------------------
#---------------------------------------------------------------

def main(data_path):
    '''
        preprocesses data for training
        args:
            data_path   =   the location of folder that contains test and train in CDB-3.1.2
    '''
    data_path=args.data_path
    train_path =os.path.join(data_path,'Train')
    test_path  =os.path.join(data_path,'Test')
    # determine number of classes
    dlist=os.listdir(train_path)
    dlist=[int(d) for d in dlist]
    dlist.sort()
    lower_thresh=dlist[0]
    upper_thresh=dlist[-1]
    nb_classes=upper_thresh-lower_thresh+1
    # log
    msg=f"Number of classes:{nb_classes}"
    LOG_INFO(msg)
    dim=64
    # processing
    # train
    #x_train , y_train   =   get_Xy(train_path,lower_thresh,nb_classes,dim=dim)
    # test
    x_test  , _    =   get_Xy(test_path,lower_thresh,nb_classes,dim=dim)   
    
    LOG_INFO(x_test.shape,mcolor='yellow')


if __name__=="__main__":
    '''
        parsing and execution
    '''
    parser = argparse.ArgumentParser("preprocessing script of CDB-3.1.2")
    parser.add_argument("--data_path", help="Path of the data folder that contains Test and Train")
    args = parser.parse_args()
    main(args)
    