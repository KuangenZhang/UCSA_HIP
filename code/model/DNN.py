# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:29:16 2019

@author: wang0918.stu
"""

import torch.nn as nn
import torch.nn.functional as F
from model.grad_reverse import grad_reverse

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=[1, 13], stride=[1,3], padding=[1, 6])
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=[1, 9], stride=[1,3], padding=[1, 4])
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, kernel_size=[1, 9], stride=[1,3], padding=[1, 4])
        self.bn3 = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, kernel_size=[1, 9], stride=[1,3], padding=[1, 4])
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=[1, 9], stride=[1,3], padding=[1, 4])
        self.bn5 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(5632, 2048)
        self.bn1_fc = nn.BatchNorm1d(2048)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x.view(x.size(0), 5632)
        x = F.relu(self.bn1_fc(self.fc1(x)))
        x = F.dropout(x, training=self.training)
        return x


class Predictor(nn.Module):
    def __init__(self, prob=0.5):
        super(Predictor, self).__init__()
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2_fc = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 7)
        self.bn_fc3 = nn.BatchNorm1d(7)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x

class DomainPredictor(nn.Module):
    def __init__(self, prob=0.5):
        super(DomainPredictor, self).__init__()
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2_fc = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 2)
        self.bn_fc3 = nn.BatchNorm1d(2)
        self.prob = prob

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = F.relu(self.bn2_fc(self.fc2(x)))
        x = self.fc3(x)
        return x