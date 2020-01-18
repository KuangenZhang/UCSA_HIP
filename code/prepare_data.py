# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:45:51 2019

@author: kuangen
"""

from utils import FileIO
from utils import utils
import mat4py as m4p
import numpy as np
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
#%% Northwestern dataset
idx_x = np.arange(0,368)
FileIO.save_mat('0_dataset/AB_156_to_186_walking.mat', is_walking = True)

#%%
x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
FileIO.load_st_AB_mat(data_path = 'data/AB_dataset/AB_', is_resize = True, 
                      leave_one_num = 1)
#%% UCI DSADS datase
# read data:[label,subjects,segments, time, sensors]
x_mat, y_mat = FileIO.read_UCI_DSADS()
FileIO.save_UCI_DSADS(x_mat, y_mat, file_path = 'data/1_dataset_UCI_DSADS/Raw/')
#%% extract features and output data
x_mat = utils.extract_UCI_features(x_mat)
FileIO.save_UCI_DSADS(x_mat, y_mat, file_path = 'data/1_dataset_UCI_DSADS/Features/')
#%% load UCI data
x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
FileIO.load_UCI_mat(data_path = 'data/1_dataset_UCI_DSADS/Features/',
                    feature_length = 6*45, is_resize = True, leave_one_num = 1)
#%%
x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
FileIO.load_UCI_mat(data_path = 'data/1_dataset_UCI_DSADS/Raw/',
                    feature_length = 125*45, is_resize = True, leave_one_num = 1)


            
            
        
