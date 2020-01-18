# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:29:32 2019

@author: kuangen
"""
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

def plot_acc_target_subject(data, image_name, sub_num, legend_vec):
    figurePath = 'images/'
    figName = figurePath + image_name
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()
    accuracy_list = []
    for i in range(sub_num):
        accuracy_list.append(data[1::2, i])
    
    plotBarForCell(accuracy_list, [],'Classification accuracy (%)',legend_vec)
    
    sub_name_list = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10']
    plt.xlim(0, 8 * sub_num)
    plt.xticks(range(4, 4 + 8 * (sub_num), 8), sub_name_list[0:sub_num])
    plt.ylim(0.65, 1)
    plt.yticks(np.arange(0.65,1.05,0.05),['65', '70', '75','80','85','90','95','100'])
    plt.savefig(figName + '.pdf', bbox_inches='tight')

def plot_acc_mean_std(data, image_name, sub_num, legend_vec):
    figurePath = 'images/'
    figName = figurePath + image_name
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 13})
    plt.tight_layout()
    accuracy_list = []
    # rows: methods cols: subjects
    for i in range(2):
        accuracy_list.append(data[i::2, :])
    
    plotErrorBarForCell(accuracy_list, [],'Classification accuracy (%)',legend_vec)
    
    sub_name_list = ['Source subjects','Target subjects']
    plt.xlim(0, np.ceil(data.shape[0] / 4) + data.shape[0])
    plt.xticks([np.ceil(data.shape[0] / 4), data.shape[0]],
                sub_name_list)
    plt.ylim(0.65, 1)
    plt.yticks(np.arange(0.65,1.05,0.05),['65', '70', '75','80','85','90','95','100'])
    plt.savefig(figName + '.pdf', bbox_inches='tight')
    
def plotErrorBarForCell(input_list,xLabel,yLabel,legend_vec):
    len_list = len(input_list)
    for i in range(len_list):
        input_vec = input_list[i]
        len_cols = len(input_vec)
        x_vec = range(i*(len_cols+2)+1, i*(len_cols+2)+len_cols+1)
        plotErrorBar(x_vec, input_vec,legend_vec)
    plt.ylabel(yLabel)


def plotBarForCell(input_list,xLabel,yLabel,legend_vec):
    len_list = len(input_list)
    for i in range(len_list):
        input_vec = input_list[i]
        len_cols = len(input_vec)
        x_vec = range((i)*(len_cols+2)+1, (i)*(len_cols+2)+len_cols+1)
        plotBar(x_vec, input_vec,legend_vec)
    plt.ylabel(yLabel)
    

def plotBar(x_vec, input_vec, legend_vec):
    len_vec = len(x_vec)
    color_vec = cm.get_cmap('Set3', 12)
    hatch_vec = ['', '-', '/', '\\', 'x', '-/','-\\','-x']
    for i in range(len_vec):
        plt.bar(x_vec[i], input_vec[i], color = color_vec(i), edgecolor='black',
                hatch = hatch_vec[i])
    plt.legend(legend_vec, loc = 'lower center', 
               ncol = len(legend_vec), bbox_to_anchor = (0.49, 1.0))
    
def plotErrorBar(x_vec, input_vec,legend_vec):
    mean_vec = np.mean(input_vec, axis = -1)
    std_vec = np.std(input_vec, axis = -1)
    len_vec = len(x_vec)
    color_vec = cm.get_cmap('Set3', 12)
    hatch_vec = ['', '-', '/', '\\', 'x', '-/','-\\','-x']
    for i in range(len_vec):
        plt.bar(x_vec[i], mean_vec[i], color=color_vec(i), edgecolor='black',
                hatch = hatch_vec[i])
    plt.errorbar(x_vec, mean_vec, yerr = std_vec, fmt='.',
                 solid_capstyle='projecting', capsize=5, color = 'black')    
    plt.legend(legend_vec, loc = 'lower center', 
               ncol = len(legend_vec), bbox_to_anchor = (0.49, 1.0))


def plot_matrix(matrix, image_name, legend_vec):
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    # # We want to show all ticks...
    ax.set(xlim = (-0.5, matrix.shape[1] - 0.5),
           ylim = (matrix.shape[0] - 0.5, -0.5),
           xticks=np.arange(matrix.shape[1]),
           yticks=np.arange(matrix.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=legend_vec.values, yticklabels=legend_vec.values
           # title='',
           # ylabel='True label',
           # xlabel='Predicted label'
           )
    # Loop over data dimensions and create text annotations.
    fmt = '.3f'
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(image_name + '.pdf', bbox_inches='tight')


# Check if data is normal distributed and calculate p values
def calc_p_matrix(data):
    data_list = []
    # rows: methods cols: subjects
    for i in range(2):
        data_list.append(data[i::2, :])
    rows = data_list[0].shape[0]
    cols = rows
    p_matrix = np.ones((rows, cols))
    is_normal_list = [True, True]
    for i in range(2):
        for r in range(rows):
            if is_normal_list[i]:
                statistic, critical_values, significance_level = stats.anderson(data[r, :])
                if statistic > critical_values[2]:
                    is_normal_list[i] = False
    for r in range(rows):
        for c in range(cols):
            if r > c:  # source domain
                p_matrix[r, c] = calc_p_value(data_list[0][r, :], data_list[0][c, :],
                                              is_normal_list[0])
            else:
                p_matrix[r, c] = calc_p_value(data_list[1][r, :], data_list[1][c, :],
                                              is_normal_list[1])
    return p_matrix, is_normal_list

def calc_one_way(data):
    data_list = []
    # rows: methods cols: subjects
    for i in range(2):
        data_list.append(data[i::2, :])
    _, p_val_source = stats.f_oneway(*list(data_list[0]))
    _, p_val_target = stats.f_oneway(*list(data_list[1]))
    return p_val_source, p_val_target




def calc_p_value(a_vec, b_vec, is_normal = True):
    if is_normal:
        _, p_val = stats.ttest_ind(a_vec, b_vec)
    else:
        _, p_val = stats.ranksums(a_vec, b_vec)
    return p_val

#def plot_all():
#%% NW dataset:
# dfs = pd.read_excel("results_leave+one+out.xlsx", sheet_name="NW-all_sensors")
# legend_vec = dfs['Method'][0::2]
# sub_num = 10
# data = dfs.values[:,2:sub_num+2].astype(np.float)
# # Fig: NW: classification accuracy for each target subject.
# plot_acc_target_subject(data, 'Fig_acc_NW_target_subject', sub_num, legend_vec)

# # Fig: NW: classification accuracy for each target subject.
# plot_acc_mean_std(data, 'Fig_acc_NW_mean_std', sub_num, legend_vec)

# Fig: Fig_p_matrix_nw
# p_matrix_nw, is_normal_list = calc_p_matrix(data)
# print(is_normal_list)
# plot_matrix(p_matrix_nw, 'images/Fig_p_matrix_nw', legend_vec)

#%% Fig: NW: mean and std for different sensors.
dfs = pd.read_excel("results_leave+one+out.xlsx", sheet_name="NW-compare_sensors")
legend_vec = dfs['Method'][0::2]
sub_num = 10
data = dfs.values[:,2:sub_num+2].astype(np.float)
# plot_acc_mean_std(data, 'Fig_acc_NW_mean_std_compare_sensors', sub_num,legend_vec)
# p_matrix_nw_sensor, is_normal_list = calc_p_matrix(data)
p_val_source_nw, p_val_target_nw = calc_one_way(data)

#%% UCI dataset:
# dfs = pd.read_excel('results_leave+one+out.xlsx', sheet_name="UCI-all_sensors")
# legend_vec = dfs['Method'][0::2]
# sub_num = 8
# data = dfs.values[:,2:sub_num+2].astype(np.float)
# # Fig: UCI: classification accuracy for each target subject.
# plot_acc_target_subject(data, 'Fig_acc_UCI_target_subject', sub_num, legend_vec)
# # Fig: UCI: classification accuracy for each target subject.
# plot_acc_mean_std(data, 'Fig_acc_NW_mean_std', sub_num, legend_vec)


# Fig: Fig_p_matrix_uci
# p_matrix_uci, is_normal_list = calc_p_matrix(data)
# print(is_normal_list)
# plot_matrix(p_matrix_uci, 'images/Fig_p_matrix_uci', legend_vec)

#%% Fig: UCI: mean and std for different sensors.
dfs = pd.read_excel("results_leave+one+out.xlsx", sheet_name="UCI-compare_sensors")
legend_vec = dfs['Method'][0::2]
sub_num = 8
data = dfs.values[:,2:sub_num+2].astype(np.float)
#plot_acc_mean_std(data, 'Fig_acc_UCI_mean_std_compare_sensors', sub_num,legend_vec)
# p_matrix_uci_sensor, is_normal_list = calc_p_matrix(data)
p_val_source_uci, p_val_target_uci = calc_one_way(data)
#plot_all()