# -*- coding: utf-8 -*-
'''
    @author: Nadim Sarwar
'''
from __future__ import print_function
#---------------------------------------------------------------
# imports
#---------------------------------------------------------------
from termcolor import colored
#---------------------------------------------------------------
def LOG_INFO(msg,mcolor='blue'):
    '''
        prints a msg/ logs an update
        args:
            msg     =   message to print
            mcolor  =   color of the msg    
    '''
    print(colored("#LOG     :",'green')+colored(msg,mcolor))
#---------------------------------------------------------------
