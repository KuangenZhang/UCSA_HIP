# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:29:16 2019

@author: wang0918.stu
"""

import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse

class Feature(nn.Module):
    def __init__(self, dataset = 'NW', sensor_num = 0):
        super(Feature, self).__init__()
        # NW, sensor_num: 33, feature_length = 12
        if 'NW' == dataset:
            sensor_count = [33, 14, 15, 4, 29, 18, 19]
            final_kernel_size = [sensor_count[sensor_num], 3]
        # UCI, sensor_num: 45, feature_length = 6
        elif 'UCI' == dataset:
            sensor_count = [45, 9, 9, 9, 9, 9]
            final_kernel_size = [sensor_count[sensor_num], 2] 
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=   [1, 1], stride=[1,1], padding=[0, 0])
        self.bn1 = nn.BatchNorm2d(32)                 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=  [1, 3], stride=[1,2], padding=[0, 1])
        self.bn2 = nn.BatchNorm2d(64)                 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=  [1, 1], stride=[1,1], padding=[0, 0])
        self.bn3 = nn.BatchNorm2d(128)                 
        self.conv4 = nn.Conv2d(128, 256, kernel_size= [1, 3], stride=[1,2], padding=[0, 1])
        self.bn4 = nn.BatchNorm2d(256)                
        self.conv5 = nn.Conv2d(256, 512, kernel_size= final_kernel_size, stride=[1,1], padding=[0, 0])    
        self.bn5 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512, 256)
        self.bn1_fc = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), stride=[1, 1], kernel_size=[1, 1])
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), stride=[1, 1], kernel_size=[1, 1])
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), stride=[1, 1], kernel_size=[1, 1], padding=0)
        # x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), stride=[1, 1], kernel_size=[1, 1], padding=0)
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), 512)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training, p = 0.5)
        return x


#class Predictor(nn.Module):
#    def __init__(self, prob=0.5):
#        super(Predictor, self).__init__()
#        self.fc1 = nn.Linear(2048, 1024)
#        self.bn1_fc = nn.BatchNorm1d(1024)
#        self.fc2 = nn.Linear(1024, 512)
#        self.bn2_fc = nn.BatchNorm1d(512)
#        self.fc3 = nn.Linear(512, 7)
#        self.bn_fc3 = nn.BatchNorm1d(7)
#        self.prob = prob
#
#    def set_lambda(self, lambd):
#        self.lambd = lambd
#
#    def forward(self, x, reverse=False):
#        if reverse:
#            x = grad_reverse(x, self.lambd)
#        x = F.relu(self.bn1_fc(self.fc1(x)))
#        x = F.dropout(x, training=self.training)
#        x = F.relu(self.bn2_fc(self.fc2(x)))
#        x = F.dropout(x, training=self.training)
#        x = self.fc3(x)
#        return x

class Predictor(nn.Module):
    def __init__(self, prob=0.5, dataset = 'NW'):
        super(Predictor, self).__init__()
        if 'NW' == dataset:
            class_num = 7
        elif 'UCI' == dataset:
            class_num = 19
#        self.fc1 = nn.Linear(256, 128)
#        self.bn1_fc = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(256, 128)
        self.bn2_fc = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, class_num)
        self.bn_fc3 = nn.BatchNorm1d(class_num)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
#        x = F.relu(self.bn1_fc(self.fc1(x)))
#        x = F.dropout(x, p = 0.5)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, p = 0.5)
        x = self.fc3(x)
        return x

class DomainPredictor(nn.Module):
    def __init__(self, prob=0.5, dataset = 'NW'):
        super(DomainPredictor, self).__init__()
        class_num = 2
        self.fc2 = nn.Linear(256, 128)
        self.bn2_fc = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, class_num)
        self.bn_fc3 = nn.BatchNorm1d(class_num)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = F.dropout(x, p = 0.5)
        x = self.fc3(x)
        return x