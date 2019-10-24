# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:53:32 2019

@author: duxiaoqin
@功能：
    （1）复现论文“2016-Variational Graph Auto-Encoders”；
"""

import os
import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from model import VGAE
import GraphReader

import common

def main():
    #删除旧实验曲线图片
    if os.path.exists(common.ResultFile):
        os.remove(common.ResultFile)
    
    #查看是否已经存在图数据文件    
    if not os.path.exists(common.NPZFile):
        coo_adjacency, edges = GraphReader.GetGraphCluster(common.DatasetFile)
        GraphReader.SaveCooAdjacency(common.NPZFile, coo_adjacency)
    else:
        coo_adjacency = GraphReader.LoadCooAdjacency(common.NPZFile)
    
    #图中节点个数    
    n_nodes = coo_adjacency.shape[0]
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True
    with tf.Session(config = config) as tf_sess:
        model = VGAE(tf_sess, n_nodes)
        model.Train(coo_adjacency)

if __name__ == '__main__':
    main()