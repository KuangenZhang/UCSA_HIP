# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:50:15 2019

@author: kuangen
"""
from utils.FileIO import *
x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            load_st_AB_mat(data_path = 'data/AB_dataset/AB_', X_dim = 2, 
            leave_one_num = 1)
print(min(y_s_train), max(y_s_train))
print(min(y_s_val), max(y_s_val))
print(min(y_s_test), max(y_s_test))
print(min(y_t_train), max(y_t_train))
print(min(y_t_val), max(y_t_val))
print(min(y_t_test), max(y_t_test))