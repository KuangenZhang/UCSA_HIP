# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:31:54 2019

@author: kuangen
"""
#%% Read data
from utils.Har_algo import *
from utils.utils import download
import numpy as np
download()
acc_traditional_nw = traditional_har(dataset='NW')
acc_traditional_uci = traditional_har(dataset = 'UCI')

# np.savetxt("results/acc_traditional_all_sensor_nw.csv",
#               acc_traditional_nw, delimiter=",")
# np.savetxt("results/acc_traditional_all_sensor_uci.csv",
#               acc_traditional_uci, delimiter=",")