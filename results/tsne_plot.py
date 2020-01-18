# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:24:06 2019

@author: kuangen
"""
import matplotlib.pyplot as plt
plt.figure(figsize=(3.5, 3.5))
for i in range(tsne.shape[0]):
    plt.text(tsne[i, 0], tsne[i, 1], str(int(labels[i])), 
             color=plt.cm.bwr(domains[i] / 1.),
             fontdict={'weight': 'bold', 'size': 9})
plt.legend(['Souce', 'Target'], loc = 'best')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
#plt.legend(['Souce', 'Target'], loc = 'lower center', 
#           ncol = 2, bbox_to_anchor = (0.49, 1.0))
plt.rcParams.update({'font.size': 9})
plt.show()
