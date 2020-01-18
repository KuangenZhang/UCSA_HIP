# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 11:34:26 2019

@author: kuangen
"""
from utils import FileIO
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
#%% LDA, no domain adaptation

def traditional_har(dataset = 'UCI'):
    if 'UCI' == dataset:
        sub_num = 8
        class_num = 19
        feature_length = 6
        sensor_num = 45
    elif 'NW' == dataset:
        sub_num = 10
        class_num = 7
    acc_s_LDA = np.zeros(sub_num)
    acc_t_LDA = np.zeros(sub_num)
    acc_s_SVM = np.zeros(sub_num)
    acc_t_SVM = np.zeros(sub_num)
    acc_s_ANN = np.zeros(sub_num)
    acc_t_ANN = np.zeros(sub_num)
    for i in range(sub_num):
        # load UCI dataset
        if 'UCI' == dataset:
            x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_UCI_mat(data_path = 'data/1_dataset_UCI_DSADS/Features/',
                            feature_length = feature_length*45, X_dim = 2, 
                            leave_one_num = i)
            x_s_train = x_s_train[:,0:feature_length*sensor_num]
            x_s_val = x_s_val[:,0:feature_length*sensor_num]
            x_s_test = x_s_test[:,0:feature_length*sensor_num]
            x_t_train = x_t_train[:,0:feature_length*sensor_num]
            x_t_val = x_t_val[:,0:feature_length*sensor_num]
            x_t_test = x_t_test[:,0:feature_length*sensor_num]
        
        # load NW dataset
        elif 'NW' == dataset:
            x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_st_AB_mat(data_path = 'data/AB_dataset/AB_', X_dim = 2, 
            leave_one_num = i)

        # print(y_s_train.shape[0] + y_s_val.shape[0] + y_s_test.shape[0],
        #       y_t_train.shape[0] + y_t_val.shape[0] + y_t_test.shape[0])
        # LDA, no domain adaptation
        clf = LDA()
        clf.fit(x_s_train, y_s_train)
        y_s_test_pred = clf.predict(x_s_test)

        start = time.clock()
        for i in range(8):
            out_prediction = clf.predict(x_s_test[[i]])
        end = time.clock()
        print('LDA: forward time for each segment:%.30f'
              % ((end - start)/8.))

        acc = accuracy_score(y_s_test, y_s_test_pred)
        print("LDA: source domain accuracy: %.2f%%" % acc)
        acc_s_LDA[i] = acc

        y_t_test_pred = clf.predict(x_t_test)
        acc = accuracy_score(y_t_test, y_t_test_pred)
        print("LDA: target domain accuracy: %.2f%%" % (acc))
        acc_t_LDA[i] = acc

        # SVM, no domain adaptation
        clf = svm.LinearSVC(max_iter=5000)
        clf.fit(x_s_train, y_s_train)
        y_s_test_pred = clf.predict(x_s_test)

        start = time.clock()
        for i in range(8):
            out_prediction = clf.predict(x_s_test[[i]])
        end = time.clock()
        print('SVM: forward time for each segment:%.30f'
              % ((end - start) / 8.))

        acc = accuracy_score(y_s_test, y_s_test_pred)
        print("SVM: source domain accuracy: %.2f%%" % acc)
        acc_s_SVM[i] = acc

        y_t_test_pred = clf.predict(x_t_test)
        acc = accuracy_score(y_t_test, y_t_test_pred)
        print("SVM: target domain accuracy: %.2f%%" % (acc))
        acc_t_SVM[i] = acc

        #%% ANN, no domain adaptation

        # load UCI dataset
        if 'UCI' == dataset:
            x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_UCI_mat(data_path = 'data/1_dataset_UCI_DSADS/Features/',
                                is_one_hot = True, is_normalized = True,
                            feature_length = feature_length*45,
                            X_dim = 2, leave_one_num = i)

        # load NW dataset
        if 'NW' == dataset:
            x_s_train, y_s_train, x_s_val, y_s_val, x_s_test, y_s_test, \
            x_t_train, y_t_train, x_t_val, y_t_val, x_t_test, y_t_test = \
            FileIO.load_st_AB_mat(data_path = 'data/AB_dataset/AB_', X_dim = 2,
                                  is_one_hot = True, is_normalized = True,
                                  leave_one_num = i)

        clf = MLPClassifier(solver='sgd', activation='tanh',learning_rate='adaptive',
                            learning_rate_init=0.1,hidden_layer_sizes=(10,class_num),
                            max_iter = 2000)
        clf.fit(x_s_train, y_s_train)
        y_s_test_pred = clf.predict(x_s_test)
        acc = accuracy_score(y_s_test, y_s_test_pred)

        start = time.clock()
        for i in range(8):
            out_prediction = clf.predict(x_s_test[[i]])
        end = time.clock()
        print('ANN: forward time for each segment:%.30f'
              % ((end - start) / 8.))

        print("ANN: source domain accuracy: %.2f%%" % acc)
        acc_s_ANN[i] = acc

        y_t_test_pred = clf.predict(x_t_test)
        acc = accuracy_score(y_t_test, y_t_test_pred)
        print("ANN: target domain accuracy: %.2f%%" % (acc))
        acc_t_ANN[i] = acc

    print ('LDA: mean of test acc in the source domain:', np.mean(acc_s_LDA))
    print ('LDA: mean of test acc in the target domain:', np.mean(acc_t_LDA))
    print ('SVM: mean of test acc in the source domain:', np.mean(acc_s_SVM))
    print ('SVM: mean of test acc in the target domain:', np.mean(acc_t_SVM))
    print ('ANN: mean of test acc in the source domain:', np.mean(acc_s_ANN))
    print ('ANN: mean of test acc in the target domain:', np.mean(acc_t_ANN))

    return np.transpose(np.c_[acc_s_LDA, acc_t_LDA, acc_s_SVM,
                              acc_t_SVM, acc_s_ANN, acc_t_ANN])