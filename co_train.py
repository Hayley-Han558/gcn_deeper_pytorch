 # -*- coding:utf-8 -*-

# http://bestzhangjin.com/2017/10/13/deeplearn/

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

import numpy as np
from numpy.linalg import inv

from sklearn.metrics import accuracy_score

from utils import *
from model import GCN

# import operator
# from functools impo rt reduce

# change ctx to mx.gpu(0) to use gpu device
ctx = mx.cpu()          # 说明是CPU
# t = 50                  # t代表取前t个标签
alpha = 1e-6            # 定义参数值————P = (L + alpha * Lambda) ** -1

if __name__ == "__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')  # 加载数据
    features = nd.array(features.toarray(), ctx = ctx)                                          # 格式转换
    y_train = nd.array(y_train, ctx = ctx)
    y_val = nd.array(y_val, ctx = ctx)

    A_tilde = adj.toarray() + np.identity(adj.shape[0])     # A+I
    # A_tilde = adj.toarray() + np.identity(adj.shape[0])     # A+I
    D = A_tilde.sum(axis = 1)                               # 度矩阵
    A_ = nd.array(np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5)), ctx = ctx)           # 对称归一化D-1/2 *A * D-1/2

    idx = np.arange(len(A_))                                # 数据预处理

    # 定义归一化操作
    Lambda = np.identity(len(A_))
    L = np.diag(D) - adj
    P = inv(L + alpha * Lambda)                             # 随机游走P = (L + alpha * Lambda) ** -1   2708*2708 归一化吸收概率矩阵？？？

    # 预处理训练集的标签集、样本，并构成字典。
    train_label = nd.argmax(y_train[idx[train_mask]], axis = 1).asnumpy()
    train_idx = set(idx[train_mask].tolist())
    train_dict = dict(zip(train_idx, map(int, train_label)))

    learningRate = 0.05
    t = int(len(idx) * learningRate)                       # 定义training_size 用idx还是 train_idx???

    print('size of training %s' % t)
    # print('training_size = %s' % len(train_idx))    # 140
    # print('training_size = %s' % len(train_label))    # 140
    # print('new dataset size: %s' % (len(train_dict)))

# -------------------------------------------定义随机游走co-training---------------------------------------
    # 计算置信度向量 随机游走
    # for k in range(y_train.shape[1]):
    #     nodes = idx[train_mask][train_label == k]                   # 每一类中
    #     probability = P[:, nodes].sum(axis = 1).flatten()           # probability 输出 是一个矩阵 1*2708
    #     for i in np.argsort(probability).tolist()[0][::-1][:t]:     # 选择前t个
    #         if i in train_dict:
    #             continue
    #         train_dict[i] = k
    #
    # # print(train_dict)
    # print('概率为：', probability)
    #         # 结点属于k类的置信度
    # print('new dataset size: %s'%(len(train_dict)))
    #
    # # 随机游走的得到的新数据集
    # new_train_index = sorted(train_dict.keys())
    # new_train_label = [train_dict[i] for i in new_train_index]

# ---------------------------仅仅使用gcn时的输入-------------------------
#     train_index = sorted(train_dict.keys())  # 对字典按照key排序
#     train_label = [train_dict[i] for i in train_index]
#     # 截取列表的前t个
#     train_index = train_index[0:t:1]
#     train_label = train_label[0:t:1]
#
#     print('train_index = ', train_index)
#     print('train_label = ', train_label)
#     # # 字典切片取前t个
#     # def dict_slice(adict, start, end):
#     #     keys = adict.keys()
#     #     dict_slice = {}
#     #     for k in keys[start:end]:
#     #         dict_slice[k] = adict[k]
#     #     return  dict_slice
#     # slice = dict_slice(train_index, 1, t)
#
#     #     print(new_train_label)
#     #     print(new_train_index)
    # 使用gcn并初始化
    net = GCN()
    net.initialize(ctx = ctx)
    net.hybridize()
# ------------------------仅仅使用GCN---------------------------

# -------------------------------------定义图卷积操作self-training-----------------------------------------
    new_G = net(features, A_)     # gcn的输出

    # 通过gcn将准确率排名
    for k in range(y_train.shape[1]):   # 按每一个类别进行处理
        nodes = idx[train_mask][train_label == k]  # 每一类
        G_result = new_G[:, nodes].sum(axis=1).flatten()    # G的大小：列表2708*1
    print('准确率排名的结果为:',G_result)

   # # 遍历G_result中的数据 输入一维列表
   #  l = []
   #  for i in range(len(G_result)):
   #      for j in range(len(G_result[i])):
   #          l.append(G_result[i][j])
   #          np.insert(a, len(l), G_result[i][j])

   #  print('-------------------print_l-----------------')
   #  print(np.array(G_result).shape)
    
    Reshape_G = G_result.reshape((1,2708))       # Reshape_G 1*2708的数组
    # print('整形后的GCN_result', np.mat(np.array(Reshape_G)))
    # print(np.argsort())

    for i in np.argsort(np.mat(np.array(Reshape_G))).tolist()[0][::-1][:t]:  # 选择前t个
        if i in train_dict:
            continue
        train_dict[i] = k

    # print('------------Gcn_train_dict-----------------')
    #     G_result = G_result.T
    #
    #     # G_result = list(G_result.T)
    #     # G_result = np.array(list(G_result.T))   # 1*2708
    #     print(G_result[0])
    #     # m = np.argsort(G_result)
    #     # print(m)
    #
    #     # # for i in np.argsort(G_result.T).tolist()[0][:t]:  # 选择前t个，筛选格式？tolist转换为列表 列表截
    #     # # for i in np.argsort(G_result[0], kind="quicksort").tolist()[:t]:
    #     # # for i in list(G_result.argsort())[:t]:
    #     #     if i in train_dict:
    #     #         continue
    #     #     train_dict[i] = k
    #
    # 自加 GCN得到的新数据集
    gcn_train_index = sorted(train_dict.keys())
    gcn_train_label = [train_dict[i] for i in gcn_train_index]
#
# '--------------------------------------------------图卷积操作结束----------------------------------------------'

    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()        # 计算交叉熵损失
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})      # 优化器

    # 开始训练
    for epoch in range(100):
        with autograd.record():
            # 返回要在“with”语句中使用的autograd记录范围上下文，并捕获需要计算梯度的代码。
            # 注意: 当使用train_mode = False转发时，对应的后退也应该使用train_mode = False，否则梯度未定义。
            output = net(features, A_)       # 调用GCN
            # output = net(features, A_)      # 把gcn的输出转化为输入

            l = loss_function(output[gcn_train_index], nd.array(gcn_train_label, ctx = ctx))
            # 读入不同的标签集和索引

        l.backward()
        trainer.step(1)
        print('training loss: %.2f'%(l.mean().asnumpy()[0]))
        
        output = net(features, A_)
        l = loss_function(output[idx[val_mask]], nd.argmax(y_val[idx[val_mask]], axis = 1))
        print('validation loss %.2f'%(l.mean().asnumpy()[0]))
        print()
    
    output = net(features, A_)
    print('testing accuracy: %.3f'%(accuracy_score(np.argmax(y_test[idx[test_mask]], axis = 1), nd.argmax(output[idx[test_mask]], axis = 1).asnumpy())))
    