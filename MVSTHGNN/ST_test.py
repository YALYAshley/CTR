from __future__ import absolute_import
from __future__ import division
import math
from typing import Tuple
# from mvpose.EasyMocap.easymocap.mytools.file_utils import select_nf
from torch.nn.parameter import Parameter
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H, hyperedge_concat
from softmax import softmax
from json2data import json2data

__all__ = ['MV_TSHL']

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
from torchreid import models
from torchreid import data_manager, metrics, lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import os.path as osp
import pandas as pd
from pandas import Series,DataFrame

from torchreid.utils.reidtools import calc_splits
from torchreid.utils.torchtools import weights_init_kaiming, weights_init_classifier

class argparse():
    pass

args = argparse()
args.epochs, args.learning_rate, args.patience = [300, 0.001, 4]
args.hidden_size, args.input_size= [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]

def gen_feature():

    '''
    窗口宽度为2s
    3个视角
    每秒5帧
    18个节点
    超图网络输入顶点数为540
    最终f_sum构成为：3（6s/2s），10帧（取2s，每秒5帧），8个动作，3个视角的同一个节点，每个节点包含（x,y,prob）
    '''
    f1 = []
    f_sum=[]
    multi_data_sum=[]
    windows = 2
    frames = 5

    data1, len_pose_file, len_views, len_lbls, _ = json2data()
    for num in range(len_views):
        multi_data = []
        for i in range(0, len(data1), len_pose_file):
            data2 = data1[i+10*num: i +10*num+ (windows * frames)]
            multi_data.append(data2)
        multi_data_sum.append(multi_data)
    # print(multi_data_sum[2])
    # print(len(multi_data_sum[2]))
    # print(len(multi_data))
    loop_num = int(len_pose_file/(windows * frames))

    for n in range(loop_num):
        f = []
        # for i in range(len_lbls):
        for joint in range(0,18):
                for view in range(0, (len_views * len_lbls)):
                    for j in range(0,(windows * frames)):
                            f1 = multi_data_sum[n][view][j]['data'][0,joint,:]
                            f.append(f1)
        f_sum.append(f)
    return f_sum

def get_train():
    f_sum = gen_feature()
    f_sum = tuple(f_sum[0])
    f_sum = torch.cat(f_sum, dim=0)
    # print(f_sum[2])
    # X = np.array(f_sum[0])
    X = f_sum.reshape(-1,540,3)
    print(X.shape)
    return X

def get_valid():
    f_sum = gen_feature()
    f_sum = tuple(f_sum[1])
    f_sum = torch.cat(f_sum, dim=0)
    # print(f_sum[2])
    # X = np.array(f_sum[0])
    X = f_sum.reshape(-1,540,3)
    print(X.shape)
    return X

        
def gen_G(X):

    G = []  

    for i in range(0, len(X)):
            H = X[i] 
            g = generate_G_from_H(H)
            G.append(g)
    print("G.size:", len(G))  
    return G

class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[1, 36, 16], class_num= 8):
        super(FrameWiseHGNN,self).__init__()
        # hgnn layers
        self.layers = nn.ModuleList()
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(hids[j - 1], hids[j]))  # set hiddens

        self.cls_layer = nn.Linear(hids[-1], class_num)  # 设置全链接层，输入格式是1*8
        print("cls_layer:",self.cls_layer)
    
    def forward(self, x):
        layers = nn.ModuleList()
        G = gen_G(x)
        f_in = torch.Tensor(x)

        for L in layers:
            f_in = L(f_in, G)  # 进入网络层
        return f_in

def train_it():

    images_train, lbls_train, images_valid, lbls_valid = gen_feature()

    net = FrameWiseHGNN()
 
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
    print_freq = 100
    num_epochs= 25

    # optim criterion
    for epoch in range(args.epochs):
        if epoch % print_freq == 0:
            print('-' * 10)
            print(f'Epoch {epoch}/{num_epochs - 1}')
        net.train()
        train_out = net(images_train)
        optimizer.zero_grad()
        loss = criterion(lbls_train,train_out)
        loss.backward()
        optimizer.step()
        train_pred = torch.argmax(train_out)

    # valid
    net.eval()
    valid_out = net(images_valid)
    loss = criterion(valid_out,lbls_valid)
    valid_pred = torch.argmax(valid_out)

    print("train_pred:",train_pred)
    print("valid_pred:",valid_pred)

    acc = metrics.accuracy(train_pred,valid_pred)
    print("accuracy:",acc)

if __name__ == '__main__':
    train_it()
