from __future__ import absolute_import
from __future__ import division
import math
from os import X_OK
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
import torch.nn.functional as F
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
from sklearn.metrics import precision_score, recall_score, f1_score

from torchreid.utils.reidtools import calc_splits
from torchreid.utils.torchtools import weights_init_kaiming, weights_init_classifier

class argparse():
    pass

args = argparse()
args.epochs, args.learning_rate, args.patience = [300, 0.001, 4]
args.hidden_size, args.input_size= [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),]

root = '/home/mn/8T/code/MVSTHGNN/data/json/'

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
    f_sum = []
    lbl_sum = []
   
    multi_data_sum=[]
    windows = 2
    frames = 5

    data1, len_pose_file, len_views, len_lbls = json2data()

# The sliding window is 2s, data——>dict
    for num in range(len_views):
        multi_data = []
        for i in range(0, len(data1), len_pose_file):
            data2 = data1[i + 10 * num: i + 10 * num + (windows * frames)]
            multi_data.append(data2)
        multi_data_sum.append(multi_data)
    # print(multi_data_sum[0])
    # print(len(multi_data_sum[2]))
    # print(len(multi_data_sum))

    loop_num = int(len_pose_file/(windows * frames))
    for n in range(loop_num):
        f = []
        lbl = []
        for i in range(len_lbls):
            for joint in range(0,18):
                for j in range(0,(windows * frames)):
                    for view in range(0, len_views):
                        f1 = multi_data_sum[n][view + 3 * i][j]['data'][0,joint,:]
                        f.append(f1)
                        lbl1 = multi_data_sum[n][view + 3 * i][j]['action']
                    lbl.append(lbl1)
            # print(lbl)
        f_sum.append(f)
        lbl_sum.append(lbl)
    # print("f.size:",len(f))
    # print("lbl.size:",lbl_sum)
    return f_sum, lbl_sum

def get_packed_data(multi_data_sum, frame_st, frame_end):
    packed_ft = None
    return packed_ft, lbl

def get_train():
    f_sum, lbl_sum = gen_feature()
    f_sum = tuple(f_sum[0])
    lbl_sum = np.array(lbl_sum[0])
    f_sum = torch.cat(f_sum, dim=0)
    # print(f_sum[2])
    # X = np.array(f_sum[0])
    X = f_sum.reshape(-1,18,3,3)
    label = lbl_sum.reshape(-1,18,1)
    # print("get_train_X:",X.shape)
    # print("get_train_label:",label.shape)
    feature_map = []
    lab_dict={'changelane': [0], 'leftturnwait': [1], 'pullover': [2], 'slowdown': [3], 'stop': [4], 'straight': [5], 'turnleft': [6], 'turnright': [7]}
    for i in range(0, len(X)):
        lbl = max(label[i])
        dic_map = {'data_X': X[i], 'lbl': lab_dict[str(lbl[0])]}
        feature_map.append(dic_map)
    # print(len(feature_map))
    return feature_map

def get_valid():
    f_sum, lbl_sum = gen_feature()
    f_sum = tuple(f_sum[1])
    lbl_sum = np.array(lbl_sum[1])
    f_sum = torch.cat(f_sum, dim=0)
    # print(f_sum[2])
    # X = np.array(f_sum[0])
    X = f_sum.reshape(-1,18,3,3)
    label = lbl_sum.reshape(-1,18,1)
    # print("get_train_X:",X)
    # print("get_train_label:",label)
    feature_map = []
    lab_dict={'changelane': [0], 'leftturnwait': [1], 'pullover': [2], 'slowdown': [3], 'stop': [4], 'straight': [5], 'turnleft': [6], 'turnright': [7]}
    for i in range(0, len(X)):
        lbl = max(label[i])
        dic_map = {'data_X': X[i], 'lbl': lab_dict[str(lbl[0])]}
        feature_map.append(dic_map)
    # print(feature_map)
    return feature_map

def gen_G(X):

    G = [] 
    # X = get_train()
    # print("len(X):",len(X))
    # for i in range(len(X)):shape(0)
    for j in range(0, len(X)): 
        H = X[j]
        # print("H:",H)
        g = generate_G_from_H(H)
        G.append(g)
    G = torch.stack((G), dim=0)
    # print("G.size:", G)
    return G
    
class MyDataSet(Dataset):
    def __init__(self,flag='train'):
        if flag=='train':
            data=get_train()
        else :
            data=get_valid()
        x=[]
        y=[]
        for i in range(len(data)):
            x.append(data[i]['data_X'])
            y.append(data[i]['lbl'])

        self.data = x
        self.label = y
        self.length =len(data)
        
    def __getitem__(self, mask):
        label = self.label[mask]
        data = self.data[mask]
        return  np.array(data),np.array(label)

    def __len__(self):
        return self.length

class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[8, 16, 1], class_num= 8):
        super(FrameWiseHGNN,self).__init__()
        self.activation = nn.Softmax(dim=-1)
        # hgnn layers
        self.layers = nn.ModuleList()  # the list of hypergraph layers
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(3, 3))  # set hiddens

        self.cls_layer = nn.Linear(18*3, class_num)  # 设置全链接层，输入格式是1*8
        print("cls_layer:",self.cls_layer)


    
    def forward(self, x):
        print("x:",x.shape)
        layers = self.layers
        G = gen_G(x)
        print("G:",G.shape)
        f_in = torch.Tensor(x)
        # f_in = np.expand_dims(f_in, axis=1)
        # f_in = torch.Tensor(f_in)
        print("f_in:",f_in.shape)

        for L in layers:
            f_in = L(f_in, G)  # 进入网络层
            print("f:",f_in.size())
        # n x c -> 1 x c
        f_in = f_in.max(1)[0]
        print("f_in.max(1)",len(f_in))
        f_in=f_in.reshape((1,18*3))
        # f_in = torch.from_numpy(f_in)
        # print("f_in.",f_in)
        # 1 x c -> 1 x 8
        out_feature = self.cls_layer(f_in)
        # print("out_feature:", out_feature)
        result = self.activation(out_feature)
        # result = softmax(out_feature)
        # print("result:", result)
        result_max = result.argmax(dim=1)
        print(result_max)
            
        return result_max

def train_it():
    train_dataset = MyDataSet(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    valid_dataset = MyDataSet(flag='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=True)
    
    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    net = FrameWiseHGNN()    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),lr=args.learning_rate)
    for epoch in range(args.epochs):
        net.train()
        train_epoch_loss = []
        for idx,(data_x,data_y) in enumerate(train_dataloader,0):
            # print(data_x)
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)
            outputs = net(data_x)
            optimizer.zero_grad()
            loss = criterion(data_y,outputs)
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx%(len(train_dataloader)//2)==0:
                print("epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader),loss.item()))
        train_epochs_loss.append(np.average(train_epoch_loss))
        
        #=====================valid============================
        net.eval()
        valid_epoch_loss = []
        for idx,(data_x,data_y) in enumerate(valid_dataloader,0):
            data_x = data_x.to(torch.float32)
            data_y = data_y.to(torch.float32)
            outputs = net(data_x)
            loss = criterion(outputs,data_y)
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
        valid_epochs_loss.append(np.average(valid_epoch_loss))
        
        #====================adjust lr========================
        lr_adjust = {
                2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
                10: 5e-7, 15: 1e-7, 20: 5e-8
            }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))
    # # images_train, lbls_train, images_valid, lbls_valid = gen_G(get_train()), gen_G(get_valid())
            
    #         train_loss = []
    #         valid_loss = []
    #         print_freq = 100
    #         num_epochs= 25
    #         train_acc = 0
    #         batch_size = 1

    #         # optim criterion
    #         for epoch in range(args.epochs):
    #             if epoch % print_freq == 0:
    #                 print('-' * 10)
    #                 print(f'Epoch {epoch}/{num_epochs - 1}')
    #             net.train()
    #             # forward
    #             train_out = net(images_train)
    #             print(len(lbl_train))
    #             #print(torch.tensor([4],dtype=torch.int8))
    #             loss = np.array(criterion(torch.tensor(train_out,dtype=torch.float32), torch.tensor(lbl_train,dtype=torch.float32)))
    #             train_loss += loss.item() * len(lbl_train)

    #             train_pred = torch.argmax(train_out)
    #             num_acc = (train_pred == lbl_train).sum()
    #             train_acc += num_acc.item()

    #             # backward
    #             optimizer.zero_grad()
    #             loss.backward()
    #             optimizer.step()
    #             print('epoch:[{}],train loss is:{:.6f},train acc is:{:.6f}'.format(epoch,
    #                                                                         train_loss / (len(images_train) * batch_size),
    #                                                                         train_acc / (len(images_train) * batch_size)))
                
                

    #         # valid
    #         net.eval()
    #         valid_epoch_loss = []
    #         valid_out = net(images_valid)
    #         loss = criterion(valid_out,lbl_valid)
    #         valid_epoch_loss.append(loss.item())
    #         valid_loss.append(loss.item())
    #         valid_pred = torch.argmax(valid_out)
    # p = precision_score(lbl_train, train_pred, average='binary')
    # r = recall_score(lbl_train, train_pred, average='binary')
    # f1score = f1_score(lbl_train, train_pred, average='binary')
    # print("train_precision:",p)
    # print("train_recall:",r)
    # print("train_f1score:",f1score)


if __name__ == '__main__':
    train_it()
    # gen_G(get_train())
    # get_train()
    # FrameWiseHGNN(hids=[1, 36, 16], class_num= 8)
