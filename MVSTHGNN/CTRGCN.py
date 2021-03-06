# -*-coding:utf-8-*-
from __future__ import absolute_import
from __future__ import division
import sys
import os.path as osp
import time
from stateofartdata import json2data, get_packed_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score,f1_score, confusion_matrix
import matplotlib.pyplot as plt

from torchreid.utils.logger import Logger
import numpy as np
import torch
from torch import nn
from CTRGCNmodel.ctrgcn import Model
from MSG3Dmodel.feeder_tools import random_shift, random_choose, auto_pading
from torch.utils.data import DataLoader, Dataset

class argparse():
    pass


args = argparse()
args.epochs, args.learning_rate, args.patience = [100, 0.1, 4]
args.device = torch.device("cuda:1")
# args.device, = [torch.device("cuda:1" if torch.cuda.is_available() else "cpu"), ]
root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/data/'
multi_data, multi_lbl = json2data(root)
import warnings
warnings.filterwarnings("ignore")

class MyDataSet(Dataset):
    def __init__(self, flag='train'):

        frame_st = 0
        frame_end = 8
        duration = (frame_end - frame_st) * 5
        self.data, self.label = get_packed_data(multi_data, multi_lbl, frame_st, frame_end)
        for i in range(duration):
            self.data.pop()
            self.label.pop()
        train_data, valid_data, train_lbl, valid_lbl = train_test_split(self.data, self.label, test_size=0.2,
                                                                        random_state=42)

        if flag == 'train':
            self.data = train_data
            self.label = train_lbl
        else:
            self.data = valid_data
            self.label = valid_lbl

        self.length = len(self.data)

    def __getitem__(self, index):

        data = self.data[index]
        label = self.label[index]
        # data = auto_pading(data, 300)
        return data, label

    def __len__(self):
        return self.length

def train_it():
    train_dataset = MyDataSet(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_dataset = MyDataSet(flag='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)

    sys.stdout = Logger(osp.join('/home/mn/8T/code/new-hgnn/MVSTHGNN/log/police_CTRlog/', 'log_{}.txt'.format(time.strftime('-%Y-%m-%d-%H-%M-%S'))))

    train_loss = []
    train_epochs_loss = []
    best_acc = 0
    best_acc_epoch = 0

    net = Model(
        num_class=9,
        num_point=18,
        num_person=2,
        graph_args=dict(),
        drop_out=0,
        adaptive=True,
        in_channels=3,
        graph='CTRGCNmodel.police.Graph').to(args.device)

    criterion = nn.CrossEntropyLoss().to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        net.train()
        train_epoch_loss = []
        train_pred_list = []
        lbl_list = []
        for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
            data = data_x.to(torch.float32).to(args.device)
            lbl = data_y.long().to(args.device)   #torch.int64
            lbl = lbl.view(lbl.size(0))

            optimizer.zero_grad()

            # compute output
            outputs = net(data)  #torch.float32
            loss = criterion(outputs, lbl)

            #backward
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            #measure accuracy and record loss
            train_pred = outputs.argmax(dim=1)
            train_pred_list.extend(train_pred.cpu().numpy().tolist())
            lbl_list.extend(lbl.cpu().numpy().tolist())

        train_pred_result = np.array(train_pred_list)
        lbl_result = np.array(lbl_list)
        acc = np.mean(train_pred_result == lbl_result)
        pre_score = precision_score(lbl_result, train_pred_result, average = 'weighted')
        rec_score = recall_score(lbl_result, train_pred_result, average = 'weighted')
        f_score = f1_score(lbl_result, train_pred_result, average='weighted')

        print(f'epoch={epoch}/{args.epochs}, loss = {loss.item():.4f}, training: acc {acc * 100:.3f}%, '
              f'precision {pre_score * 100:.3f}%, recall {rec_score:.5f}, f1 {f_score:.5f}')


        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        net.eval().to(args.device)
        valid_pred_list = []
        lbl_valid_list = []
        with torch.no_grad():
            for idx, (data_x, data_y) in enumerate(valid_dataloader, 0):
                data_valid = data_x.to(torch.float32).to(args.device)
                lbl_valid = data_y.long().to(args.device)  # torch.int64
                lbl_valid = lbl_valid.view(lbl_valid.size(0))

                # compute output
                outputs = net(data_valid)

                valid_pred = outputs.argmax(dim=1)
                valid_pred_list.extend(valid_pred.cpu().numpy().tolist())
                lbl_valid_list.extend(lbl_valid.cpu().numpy().tolist())

        valid_pred_result = np.array(valid_pred_list)
        lbl_valid_result = np.array(lbl_valid_list)
        valid_acc = np.mean(valid_pred_result == lbl_valid_result)
        pre_valid_score = precision_score(lbl_valid_result, valid_pred_result, average='weighted')
        rec_valid_score = recall_score(lbl_valid_result, valid_pred_result, average='weighted')
        f_valid_score = f1_score(lbl_valid_result, valid_pred_result, average='weighted')

        if valid_acc > best_acc:
            best_acc = valid_acc
            best_acc_epoch = epoch + 1

        print(f"epoch={epoch}/{args.epochs}, validing: acc {valid_acc * 100:.3f}%, "
              f"precision {pre_valid_score * 100:.3f}%, recall {rec_valid_score:.5f}, f1 {f_valid_score:.5f}")
        print(f"Best accuracy: {best_acc * 100:.3f}%, Epoch number: {best_acc_epoch}")
    plt.clf()

        # # ====================adjust lr========================
        # lr_adjust = {
        #     2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
        #     10: 5e-7, 15: 1e-7, 20: 5e-8
        # }
        #
        # if epoch in lr_adjust.keys():
        #     lr = lr_adjust[epoch]
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        #     print('Updating learning rate to {}'.format(lr))



if __name__ == '__main__':
    train_it()