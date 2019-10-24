# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 16:32:10 2019

@author: duxiaoqin
@功能：
    （1）复现论文“2016-Variational Graph Auto-Encoders”；
    （2）igraph绘图需要pycairo包；
"""

import numpy as np
import scipy.sparse as sp
import igraph as ig
import os

import common

#filename文件格式：每行2个整型，表示边信息（节点索引-节点索引）
#读取filename，建立图结构
def ReadGraphFile(filename):
    #获取图的节点个数
    with open(filename, 'r') as file:
        indexes = list(map(int, file.read().split()))
        n_nodes = max(indexes) + 1#节点个数
    
    G = ig.Graph()#图对象
    G.add_vertices(n_nodes)
    
    for i_edge in range(len(indexes)//2):
        G.add_edge(indexes[2 * i_edge], indexes[2 * i_edge + 1])#添加边

    return G

def GetGraphCluster(filename, idx = -2):
    #从文件中读取数据，生成图结构
    G = ReadGraphFile(filename)
    
    #G.vcount返回节点个数，G.vs["name"]是一个列表，存放每个节点的名称
    G.vs["name"] = list(range(G.vcount()))
    
    #过滤掉一些零散的孤立节点
    #提取整块图用作训练数据集
    if idx == -2:#缺省情况下，G.components()返回强连通分量，giant()返回它的giant community
        G = G.components().giant()
    if idx == -1:
        G = G.components().giant()
        G = G.community_multilevel().giant()       
    else:       
        com = G.community_multilevel()
        for i in range(com.__len__()) :
            if idx in com.subgraph(i).vs["name"]:
                G = com.subgraph(i)
                break
    #保存整块图
    ig.plot(G)
    ig.plot(G, target = './data/strong_components_' + common.DatasetName + '.pdf')
    
    #获取图的边列表
    edges = G.get_edgelist()
    #图的节点数
    n_nodes = G.vcount()
    row = []
    col = []
    data = []
    for edge in edges:
        #邻接矩阵具有对称性
        row.extend([edge[0], edge[1]])#extend以可迭代的方式（列表元素打散）将元素放入列表中
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])
    #提供'IJV' or 'Triplet'（即data、row、col）生成稀疏邻接矩阵
    coo_adjacency = sp.coo_matrix((data, (row, col)), shape = (n_nodes, n_nodes))

    return coo_adjacency, edges#返回稀疏邻接矩阵和边列表

def SaveCooAdjacency(filename, coo_adjacency):
    #将指定数组变量保存到.npz（未压缩格式）文件中，关键字名称就是相应的变量名称
    np.savez(filename, data = coo_adjacency.data, row = coo_adjacency.row, col = coo_adjacency.col, shape = coo_adjacency.shape)

def LoadCooAdjacency(filename):
    #读取.npz文件，将保存的变量读到内存中
    loader = np.load(filename)

    return sp.coo_matrix((loader['data'], (loader['row'],loader['col'])), shape=loader['shape'])

#本函数内部对coo_adjacency做了修改，但不会影响到外界实参
def GetNormAdjacencyInfo(coo_adjacency):
    #邻接矩阵的每个对角元素设置为1（节点连接自身）——但是，也需要考虑节点已经自连的情况（后续改进）
    coo_adjacency = coo_adjacency + sp.eye(coo_adjacency.shape[0])
    #计算每个节点的出度，并且将矩阵类型转换成数组类型
    degree = np.array(coo_adjacency.sum(1))
    #出度矩阵D^(-1/2)，d_inv为对角矩阵类型
    d_inv = sp.diags(np.power(degree, -0.5).flatten())
    #利用D与A的对称性计算归一化邻接矩阵D^(-1/2)AD^(-1/2)
    #normalized类型为csc_matrix（还有一种类型为csr_matrix，支持行切片）
    normalized = coo_adjacency.dot(d_inv).transpose().dot(d_inv)
    
    return GetAdjacencyInfo(sp.coo_matrix(normalized))

#获取邻接矩阵的信息（row、col与对应的value）
#coo_adjacency是以坐标形式表示的邻接矩阵：(data, (row, col))，其中data, row, col均为一维数组
def GetAdjacencyInfo(coo_adjacency):
    #垂直方式堆叠row和col数组，然后转置，形成一个NX2索引数组（ndarray）
    #row、col为一维数组（ndarray）
    edges = np.vstack((coo_adjacency.row, coo_adjacency.col)).transpose()
    #对应的数据（一维数组ndarray）
    values = coo_adjacency.data

    return edges, values

#coo_adjacency为sparse.coo_matrix类型
def SplitTrainTestDataset(coo_adjacency):
    n_nodes = coo_adjacency.shape[0]
    #返回上三角矩阵（k=1>0，不包括主对角线元素），稀疏格式
    #邻接矩阵是实对称矩阵，这种操作是合理的
    coo_adjacency_upper = sp.triu(coo_adjacency, k = 1)
    #获取它的邻接信息
    edges, values = GetAdjacencyInfo(coo_adjacency_upper)
    
    #测试数据集和验证数据集各占比10%，edges维度为NX2，其中N为边个数
    n_tests = int(np.floor(edges.shape[0]/10.))
    n_valids = int(np.floor(edges.shape[0]/10.))

    #所有边的索引
    idx_all = list(range(edges.shape[0]))
    #打乱边索引
    np.random.shuffle(idx_all)
    #用于测试的边索引
    idx_tests = idx_all[:n_tests]
    #用于验证的边索引
    idx_valids = idx_all[n_tests : (n_tests + n_valids)]

    test_edges = edges[idx_tests]
    valid_edges = edges[idx_valids]
    #沿着行，删除指定索引（测试边和验证边）的边，留下训练（边）数据集
    train_edges = np.delete(edges, idx_tests + idx_valids, axis = 0)
    
    #存放原图数据集中不存在的边（反例），用于测试与验证
    #反例边数与正例边数一样
    test_edges_neg = []
    valid_edges_neg = []
    
    while (len(test_edges_neg) < len(test_edges)):#与测试正例边数一样
        #随机生成边
        n1 = np.random.randint(0, n_nodes)
        n2 = np.random.randint(0, n_nodes)
        if n1 == n2:
            continue
        #与上三角边索引一致：节点1索引<节点2索引        
        if n1 < n2:
            edge_to_add = [n1, n2]
        else:
            edge_to_add = [n2, n1]
        
        #edges类型为ndarray，维度为NX2
        #edges[:]==edge_to_add的类型为numpy.ndarray，维度为NX2
        #numpy数组的all方法，检查指定axis的所有元素是否为True，此处检查（每行的）2列
        if any((edges[:] == edge_to_add).all(1)):#随机生成的边已经存在于训练边数据集中
            continue
        test_edges_neg.append(edge_to_add)#添加不存在的边，用于测试
        
    while (len(valid_edges_neg) < len(valid_edges)):#与验证正例边数一样
        n1 = np.random.randint(0, n_nodes)
        n2 = np.random.randint(0, n_nodes)
        if n1 == n2:
            continue        
        if n1 < n2:
            edge_to_add = [n1, n2]
        else:
            edge_to_add = [n2, n1]
            
        if any((edges[:] == edge_to_add).all(1)):
            continue
        valid_edges_neg.append(edge_to_add)
    
    #生成训练数据集    
    row = []
    col = []
    data = []
    for edge in train_edges:
        row.extend([edge[0], edge[1]])
        col.extend([edge[1], edge[0]])
        data.extend([1, 1])
    train_coo_adjacency = sp.coo_matrix((data, (row,col)), shape=(n_nodes, n_nodes))

    return train_coo_adjacency, test_edges, test_edges_neg, valid_edges, valid_edges_neg