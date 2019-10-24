# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:14:15 2019

@author: duxiaoqin
@功能：
    （1）复现论文“2016-Variational Graph Auto-Encoders”；
"""

DatasetName = 'citation'
#DatasetName = 'facebook'

DatasetFile = './data/' + DatasetName + '.txt'
NPZFile = './data/sparse_graph_' + DatasetName + '.npz'
ResultFile = './result/ROC_curve_' + DatasetName + '.png'