# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:17:08 2019

@author: duxiaoqin
@功能：
    （1）复现论文“2016-Variational Graph Auto-Encoders”；
"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve

import GraphReader
import utils

import common

class VGAE(object):
    def __init__(self, tf_sess, n_nodes):
        self.tf_sess = tf_sess#Tensorflow进程
        self.n_nodes = n_nodes#图结构中节点个数

        self.n_hiddens = 200#图自编码器中编码器的隐单元个数
        self.n_embeddings = 128#隐编码单元个数
        self.dropout = True
        self.learning_rate = 0.05
        self.epochs = 100
                   
        self.shape = np.array([self.n_nodes, self.n_nodes])#邻接表模版（空数组）

        #（稀疏）邻接表
        self.tf_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape, name='tf_sparse_adjacency')
        #（稀疏）归一化邻接表
        self.tf_norm_sparse_adjacency = tf.sparse_placeholder(tf.float32, shape=self.shape, name='tf_norm_sparse_adjacency')
        self.keep_prob = 0.5 #在进行Dropout操作时，保留神经元的概率
        
        self.sigmoid = np.vectorize(utils.sigmoid)#向量化sigmoid函数
        
        #构造VGAE
        self.__BuildVGAE()
        
    def __BuildVGAE(self):
        #定义变分编码器计算节点
        self.TFNode_VEncoder()
        #定义变分解码器计算节点
        #注意：该节点输出没有使用激活函数，原因是
        #tf.nn.weighted_cross_entropy_with_logits中包含激活函数的调用
        tfnode_raw_adjacency_pred = self.TFNode_VDecoder()
        
        #定义KL散度D(q(z)||p(z))计算节点
        self.tfnode_latent_loss = -(0.5 / self.n_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * tf.log(self.tfnode_sigma) - tf.square(self.tfnode_mu) - tf.square(self.tfnode_sigma), 1))
        
        #将稀疏矩阵转换成常规矩阵
        #validate_indices = False，不检测索引有序存放和重复性；否则，检测上述2项
        tf_dense_adjacency = tf.reshape(tf.sparse_tensor_to_dense(self.tf_sparse_adjacency, validate_indices=False), self.shape)
        #定义交叉熵权重计算节点（引用公式见后）
        tfnode_w1 = (self.n_nodes * self.n_nodes - tf.reduce_sum(tf_dense_adjacency)) / tf.reduce_sum(tf_dense_adjacency)
        #定义重构误差权重计算节点（引用公式见后）
        tfnode_w2 = self.n_nodes * self.n_nodes / (self.n_nodes * self.n_nodes - tf.reduce_sum(tf_dense_adjacency))
        
        #定义重构误差计算节点
        #重构误差：q(Z|X)与P(X|Z)之间的交叉熵
        #tensorflow带权值交叉熵，计算公式如下：
        #targets * -log(sigmoid(logits)) * pos_weight + (1 - targets) * -log(1 - sigmoid(logits))
        self.tfnode_reconst_loss =  tfnode_w2 * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets = tf_dense_adjacency, logits = tfnode_raw_adjacency_pred, pos_weight = tfnode_w1))
        #定义总体误差函数计算节点
        #误差函数=重构误差+KL散度
        self.tfnode_all_loss = self.tfnode_reconst_loss + self.tfnode_latent_loss

        #定义tensorflow优化器与（Adam梯度下降训练）计算节点
        self.tf_optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.tf_optimizer_minimize = self.tf_optimizer.minimize(self.tfnode_all_loss)

        #初始化tf计算图（注意，所有的tf.Variable是异步初始化的，它们之间不要存在依赖关系）
        tf_init = tf.global_variables_initializer()
        self.tf_sess.run(tf_init)

    def TFNode_VEncoder(self):
        #原论文：均值与对数标准方差网络共享第1层权值。实验表明，精度提高了5%
        self.W0 = utils.UniformRandomWeights(shape = [self.n_nodes, self.n_hiddens])

        #均值网络的权值与偏置：2层GCN（图卷积网络）
        self.mu_B0 = tf.Variable(tf.constant(0.01, dtype = tf.float32, shape = [self.n_hiddens]))
        self.mu_W1 = utils.UniformRandomWeights(shape = [self.n_hiddens, self.n_embeddings])
        self.mu_B1 = tf.Variable(tf.constant(0.01, dtype = tf.float32, shape = [self.n_embeddings]))
        
        #对数标准方差网络的权值与偏置：2层GCN（图卷积网络）
        self.sigma_B0 = tf.Variable(tf.constant(0.01, dtype = tf.float32, shape = [self.n_hiddens]))
        self.sigma_W1 = utils.UniformRandomWeights(shape = [self.n_hiddens, self.n_embeddings])
        self.sigma_B1 = tf.Variable(tf.constant(0.01, dtype = tf.float32, shape = [self.n_embeddings]))

        #2层GCN计算均值
        #计算第一层GCN的输出：ReLU(AW+b)
        tfnode_mu_hidden0 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.mu_B0)
        if self.dropout:#应用Dropout
            tfnode_mu_hidden0_dropout = tf.nn.dropout(tfnode_mu_hidden0, self.keep_prob)
        else:
            tfnode_mu_hidden0_dropout = tfnode_mu_hidden0

        #计算第二层GCN的输出：AhW+b，不带激活函数
        self.tfnode_mu = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency, tfnode_mu_hidden0_dropout, self.mu_W1, self.mu_B1)
        
        #方差网络（准确地说，对数标准差）：2层GCN（图卷积网络）
        #计算第一层GCN的输出：ReLU(AW+b)
        tfnode_sigma_hidden0 = utils.FirstGCNLayerWithActiveFun_NoX(self.tf_norm_sparse_adjacency, self.W0, self.sigma_B0)
        if self.dropout:
            tfnode_sigma_hidden0_dropout = tf.nn.dropout(tfnode_sigma_hidden0, self.keep_prob)
        else:
            tfnode_sigma_hidden0_dropout = tfnode_sigma_hidden0
            
        #计算第二层GCN的输出：AhW+b，不带激活函数
        tfnode_log_sigma = utils.SecondGCNLayerWithoutActiveFun(self.tf_norm_sparse_adjacency, tfnode_sigma_hidden0_dropout, self.sigma_W1, self.sigma_B1)
        self.tfnode_sigma = tf.exp(tfnode_log_sigma)#还原为标准差

        #隐变量采样（均值与标准方差）
        return utils.GaussianSampleWithTF(self.tfnode_mu, self.tfnode_sigma)

    def TFNode_VDecoder(self):
        #隐变量采样（均值与标准方差）
        z = utils.GaussianSampleWithTF(self.tfnode_mu, self.tfnode_sigma)
        #重构邻接矩阵A=sigmoid(ZZ^T)，此处没有使用激活函数，
        #因为tensorflow.nn.weighted_cross_entropy_with_logits函数中包括了该激活函数的映射，
        #参见__BuildVGAE成员函数。
        adjacency_pred = tf.matmul(z, z, transpose_a = False, transpose_b = True)

        return adjacency_pred

    def Train(self, coo_adjacency):
        #将数据集划分成训练集、测试集与验证集
        train_coo_adjacency, test_edges, test_edges_neg, valid_edges, valid_edges_neg = GraphReader.SplitTrainTestDataset(coo_adjacency)
            
        #获取邻接矩阵与归一化邻接矩阵的信息：边集及其值
        edges, values = GraphReader.GetAdjacencyInfo(train_coo_adjacency)
        norm_edges, norm_values = GraphReader.GetNormAdjacencyInfo(train_coo_adjacency)

        #将邻接矩阵与归一化矩阵信息（边集及其值）输入给VGAE，训练网络
        feed_dict = {self.tf_sparse_adjacency : (edges, values), self.tf_norm_sparse_adjacency : (norm_edges, norm_values)}

        for i in range(self.epochs):
            #提供输入数据，运行指定计算图
            minimizer, loss, latent_loss, reconst_loss, self.mu_output, self.sigma_output = self.tf_sess.run([self.tf_optimizer_minimize, self.tfnode_all_loss, self.tfnode_latent_loss, self.tfnode_reconst_loss, self.tfnode_mu, self.tfnode_sigma], feed_dict = feed_dict)
            if i % 10 == 0:
                auc, ap = self.CalcAUC_AP(test_edges, test_edges_neg)
                print("At step {0} \n Loss: {1} \n Average Precision: {2}.".format(i, loss, ap))
                
        fpr, tpr, tresholds = self.CalcROC_Curve(test_edges, test_edges_neg)

        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC Curve for ' + common.DatasetName)
        plt.show()
        fig.savefig(common.ResultFile)

    #使用numpy函数生成指定均值与方差的隐编码（高斯分布随机数）
    def SampleLatentCode(self):
        z = utils.GaussianSampleWithNP(self.mu_output, self.sigma_output)
        return z
    
    #生成隐编码，使用sigmoid(ZZ^T)预测邻接矩阵（参见论文第2页）
    def PredictAdjacency(self):
        z = self.SampleLatentCode()
        adjacency_pred = np.dot(z, np.transpose(z))
        return self.sigmoid(adjacency_pred)
    
    #AUC：Area Under the ROC Curve
    #AP: Average Precision
    def CalcAUC_AP(self, pos_edges, neg_edges):
        #随机生成隐编码，并使用它生成（预测）邻接矩阵
        adjacency_pred = self.PredictAdjacency()
        
        y_scores = []
        
        for edge in pos_edges:#正例边（存在的边）
            y_scores.append(adjacency_pred[edge[0], edge[1]])
            
        for edge in neg_edges:#反例边（训练图结构中不存在的边）
            y_scores.append(adjacency_pred[edge[0], edge[1]])
            
        #真实边标记（1：存在，0：不存在）
        y_trues = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
        
        auc_score = roc_auc_score(y_trues, y_scores)#sklearn.metrics函数，统计AUC
        ap_score = average_precision_score(y_trues, y_scores)#sklearn.metrics函数，统计AP

        return auc_score, ap_score

    #计算ROC曲线数据
    def CalcROC_Curve(self, pos_edges, neg_edges):
        adjacency_pred = self.PredictAdjacency()
        
        y_scores = []
        
        for edge in pos_edges:
            y_scores.append(adjacency_pred[edge[0], edge[1]])
        
        for edge in neg_edges:
            y_scores.append(adjacency_pred[edge[0], edge[1]])
        
        y_trues = np.hstack([np.ones(len(pos_edges)), np.zeros(len(neg_edges))])
        
        fpr, tpr, tresholds = roc_curve(y_trues, y_scores)#sklearn.metrics函数

        return fpr, tpr, tresholds