from __future__ import absolute_import
from __future__ import division
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H, hyperedge_concat
from json2data import json2data
from sklearn.metrics import precision_score, recall_score,f1_score

# __all__ = ['MV_TSHL']

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

device = "cpu"

class argparse():
    pass


args = argparse()
args.epochs, args.learning_rate, args.patience = [100, 0.001, 4]
args.hidden_size, args.input_size = [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), ]
train_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/valid/json/'
valid_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/valid/json/'
multi_train_data, train_lbls = json2data(train_root)
multi_valid_data, valid_lbls = json2data(valid_root)

def get_packed_data(multi_data_sum, multi_lbls, frame_st, frame_end):
    """

    Args:
        multi_data_sum:  all, 3_view, 18_joints, xyp--> all,  3, 18, 3
        multi_lbls:  all, 1
        frame_st:
        frame_end:

    Returns:
        packed_multi_data:  10((frame_end - frame_st) * 5), 3_view, 18_joints, 3(xyp)
        packed_multi_lbl:
    """

    packed_multi_data = []
    packed_multi_lbl = []
    interval_frame = 2
    duration = (frame_end - frame_st) * 5

    for i in range(0, len(multi_data_sum), interval_frame):
        data2 = multi_data_sum[i : i + duration]
        lbl = multi_lbls[i]
        packed_multi_data.append(data2)
        packed_multi_lbl.append(lbl)

    return packed_multi_data, packed_multi_lbl

class MyDataSet(Dataset):
    def __init__(self, flag='train'):
        if flag == 'train':
            frame_st = 0
            frame_end = 2
            duration = (frame_end - frame_st) * 5
            self.data, self.label = get_packed_data(multi_train_data, train_lbls, frame_st, frame_end)
            for i in range(duration):
                self.data.pop()
                self.label.pop()
        else:
            frame_st = 0
            frame_end = 2
            duration = (frame_end - frame_st) * 5
            self.data, self.label = get_packed_data(multi_valid_data, valid_lbls, frame_st, frame_end)
            for i in range(duration):
                self.data.pop()
                self.label.pop()

        self.length = len(self.data)

    def __getitem__(self, index):

        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return self.length

def gen_batch_G(X):
    """

    Args:
        X: bz, [n_frame, 3_view, 18joints, 3]

    Returns:
        batch_G: bz * n * n
    """
    batch_G = []
    for _idx in range(X.size(0)):
        H = None
        cur_x =X[_idx]
        # gen G: n * n
        cur_G = get_feature(cur_x)
        packed_cur_G = tuple(cur_G)
        packed_cur_G = torch.cat(packed_cur_G, dim=0)
        # bz * n_frame * 3_view * 18 * xyp ---->> bz * n * c
        ft = packed_cur_G.reshape(-1, 3)
        H_spatial = construct_H_with_KNN(ft, K_neigs=3,
                                         is_probH=True,
                                         m_prob=1, mode="global", total_node_num=18, S=3)
        H = hyperedge_concat(H, H_spatial)
        g = generate_G_from_H(H)
        batch_G.append(g)
    batch_G = torch.stack(batch_G)
    return batch_G # bz * n * n

def get_feature(data):
    """

    Args:
        data: from gen_batch_G(X), [n_frame, 3_view, 18joints, 3]

    Returns: spatial generate
        f: [n_frame, 3_view, 18joints, 3]
    """
    f = []
    for i in range(len(data)):
        for joint in range(len(data[0][0])):  # 18 joints
            for view in range(len(data[0])):  # 3 views
                X = data[i][view][joint]
                f.append(X)
    return f

# def gen_G(X):
#     """
#
#     Args:
#         X: bz * n * c
#
#     Returns:
#         G: bz * n * m
#     """
#     G = []
#     X = gen_batch_G(X)
#     for feature in X:  # iter over each batch
#         for ft in feature:
#             H = None
#             ft = tuple(ft)
#             ft = torch.cat(ft, dim=0)
#             ft = ft.reshape(-1, n, 3)
#             H_spatial = construct_H_with_KNN(ft, K_neigs=3,
#                                              is_probH=True,
#                                              m_prob=1, mode="global", total_node_num=len(feature), S=3)
#             H = hyperedge_concat(H, H_spatial)
#             g = generate_G_from_H(H)
#         G.append(g)
#     G = torch.stack((G), dim=0)
#     return G

class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[3, 32, 32], class_num=8):
        super(FrameWiseHGNN, self).__init__()
        self.activation = nn.Softmax(dim=1)
        # hgnn layers
        self.layers = nn.ModuleList()  # the list of hypergraph layers
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(hids[j - 1], hids[j]))  # set hiddens
        self.cls_layer = nn.Linear(hids[-1], class_num)  # 设置全链接层，输入格式是1*8
        print("cls_layer:", self.cls_layer)

    def forward(self, x, G):

        for L in self.layers:
            x = L(x, G)  # 进入网络层
        # bz * n * c -> bz * c
        x = x.max(1)[0]
        # 1 x c -> 1 x 8
        out_feature = self.cls_layer(x)

        # result = self.activation(out_feature)
        result = out_feature
        return result


def train_it():
    train_dataset = MyDataSet(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=False)
    valid_dataset = MyDataSet(flag='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=False)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []

    all_cnt = 0
    net = FrameWiseHGNN()
    # criterion = nn.NLLLoss()
    pre_score = 0
    rec_score = 0
    f_score = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        net.train()
        train_epoch_loss = []
        acc_value = []
        for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
            data = data_x.to(torch.float32).to(device)
            lbl = data_y.long().to(device)   #torch.int64
            lbl = lbl.view(lbl.size(0))

            # bz * n_frame * 3_view * 18 * xyp ---->> bz * n * c
            data_input = data.view(data.size(0), -1, 3)
            G = gen_batch_G(data)

            optimizer.zero_grad()

            # compute output
            outputs = net(data_input, G)  #torch.float32
            loss = criterion(outputs, lbl)

            #backward
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())

            #measure accuracy and record loss
            train_pred = outputs.argmax(dim=1)
            acc = torch.mean((train_pred == lbl).float())
            acc_value.append(acc.data.item())
            pre_score = precision_score(lbl, train_pred, average = 'weighted')
            rec_score = recall_score(lbl, train_pred, average = 'weighted')
            f_score = f1_score(lbl, train_pred, average='weighted')

            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}, mean training acc is:{:.3f}%, precision is :{:.3f}%, recall is :{:.3f}, f1 is :{:.3f}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item(), np.mean(acc_value)*100, pre_score * 100, rec_score, f_score))
        train_epochs_loss.append(np.average(train_epoch_loss))


        # =====================valid============================
        net.eval()
        valid_epoch_loss = []
        valid_acc_value = []
        pre_valid_score = 0
        rec_valid_score = 0
        f_valid_score = 0
        for idx, (data_x, data_y) in enumerate(valid_dataloader, 0):
            data = data_x.to(torch.float32).to(device)
            lbl = data_y.long().to(device)  # torch.int64
            lbl = lbl.view(lbl.size(0))

            data_input = data.view(data.size(0), -1, 3)
            G = gen_batch_G(data)

            optimizer.zero_grad()

            # compute output
            outputs = net(data_input, G)
            loss = criterion(outputs, lbl)

            valid_pred = outputs.argmax(dim=1)
            acc = torch.mean((valid_pred == lbl).float())
            valid_acc_value.append(acc.data.item())
            pre_valid_score = precision_score(data_y, valid_pred, average='weighted')
            rec_valid_score = recall_score(data_y, valid_pred, average='weighted')
            f_valid_score = f1_score(data_y, valid_pred, average='weighted')

            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            if idx % (len(valid_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}, mean validing acc is:{:.3f}%, precision is :{:.3f}%, recall is :{:.3f}, f1 is :{:.3f}".format(epoch, args.epochs, idx, len(valid_dataloader), loss.item(), np.mean(valid_acc_value) * 100, pre_valid_score * 100, rec_valid_score,f_valid_score))
        valid_epochs_loss.append(np.average(valid_epoch_loss))

        # ====================adjust lr========================
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Updating learning rate to {}'.format(lr))


if __name__ == '__main__':
    train_it()