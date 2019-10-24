# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 17:14:25 2019

@author: duxiaoqin
@功能：
    （1）复现论文“2016-Variational Graph Auto-Encoders”；
"""

import tensorflow as  tf
import numpy as np

#按指定范围随机初始化浮点型变量，并生成Tensorflow变量
def UniformRandomWeights(shape, name = None):
    randoms = tf.random_uniform(shape, minval = -np.sqrt(6.0 / (shape[0] + shape[1])), maxval = np.sqrt(6.0 / (shape[0] + shape[1])), dtype=tf.float32)
    return tf.Variable(randoms, name = name)#返回Tensorflow变量

#返回指定均值与方差的高斯分布（调用Tensorflow函数）
def GaussianSampleWithTF(mean, diag_cov):
    z = mean + tf.random_normal(diag_cov.shape) * diag_cov
    return z

#返回指定均值与方差的高斯分布（调用numpy函数）
def GaussianSampleWithNP(mean, diag_cov):
    z = mean + np.random.normal(size = diag_cov.shape) * diag_cov
    return z

#计算第一层GCN的输出：ReLU(AW+b)
#注意，没有使用输入特征X，对应论文中的VGAE*配置
def FirstGCNLayerWithActiveFun_NoX(norm_adj_mat, W, b):
    return tf.nn.relu(tf.add(tf.sparse_tensor_dense_matmul(norm_adj_mat, W), b))

#计算第二层GCN的输出：AhW+b，不带激活函数
def SecondGCNLayerWithoutActiveFun(norm_adj_mat, h, W, b):
    return tf.add(tf.matmul(tf.sparse_tensor_dense_matmul(norm_adj_mat, h), W), b)

#sigmoid函数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))