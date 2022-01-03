from __future__ import absolute_import
from __future__ import division
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H, hyperedge_concat
from json2data import json2data

# __all__ = ['MV_TSHL']

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

device = "cpu"

class argparse():
    pass


args = argparse()
args.epochs, args.learning_rate, args.patience = [300, 0.001, 4]
args.hidden_size, args.input_size = [40, 30]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), ]
# train_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/train/json/'
valid_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/valid/json/'
# multi_train_data, train_lbls = json2data(train_root)
multi_valid_data, valid_lbls = json2data(valid_root)

def get_packed_data(multi_data_sum, multi_lbls, frame_st, frame_end):
    # packed_ft = None
    packed_multi_data = []
    packed_multi_lbl = []
    interval_frame = 2
    duration = (frame_end - frame_st) * 5

    for i in range(0, len(multi_data_sum), interval_frame):
        data2 = multi_data_sum[i : i + duration]
        lbl = multi_lbls[i : i + duration]
        packed_multi_data.append(data2)
        packed_multi_lbl.append(lbl)

    return packed_multi_data, packed_multi_lbl


# def get_train():
#
#     f_sum, lbl_sum = get_packed_data(multi_train_data, train_lbls, 0, 1)
#     for i in range(len(f_sum)):
#         for j in range(len(f_sum[0])):
#         f_sum = tuple(f_sum)
#         lbl_sum = np.array(lbl_sum)
#         f_sum = torch.cat(f_sum, dim=0)
#
#         X = f_sum.reshape(-1, 18, 3, 3)
#         label = lbl_sum.reshape(-1, 18, 1)
#
#     feature_map = []
#     for i in range(0, len(X)):
#         lbl = max(label[i])
#         dic_map = {'data_X': X[i], 'lbl': lab_dict[str(lbl[0])]}
#         feature_map.append(dic_map)
#     # print(len(feature_map))
#     return feature_map

def get_valid():

    f_sum, lbl_sum = get_packed_data(multi_valid_data, valid_lbls, 0, 1)
    for i in range(len(f_sum)):
        for j in range(len(f_sum[0])):  #5
            for view in range(len(f_sum[0][0])):  #3 views
                for joint in range(len(f_sum[0][0][0])):   # 18 joints
                    X = f_sum[i][j][view][joint]


            f_sum = tuple(f_sum)
            lbl_sum = np.array(lbl_sum)
            f_sum = torch.cat(f_sum, dim=0)
            # print(f_sum[2])
            # X = np.array(f_sum[0])
            X = f_sum.reshape(-1, 18, 3, 3)
            label = lbl_sum.reshape(-1, 18, 1)
    # print("get_train_X:",X.shape)
    # print("get_train_label:",label.shape)
    feature_map = []
    lab_dict = {'changelane': [0], 'leftturnwait': [1], 'pullover': [2], 'slowdown': [3], 'stop': [4], 'straight': [5],
                'turnleft': [6], 'turnright': [7]}
    for i in range(0, len(X)):
        lbl = max(label[i])
        dic_map = {'data_X': X[i], 'lbl': lab_dict[str(lbl[0])]}
        feature_map.append(dic_map)
    # print(len(feature_map))
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
    def __init__(self, flag='train'):
        if flag == 'train':
            data = get_train()
        else:
            data = get_valid()
        x = []
        y = []
        for i in range(len(data)):
            x.append(data[i]['data_X'])
            y.append(data[i]['lbl'])
        self.data = x
        self.label = y
        self.length = len(data)

    def __getitem__(self, index):
        label = self.label[index]
        data = self.data[index]
        return np.array(data), label

    def __len__(self):
        return self.length


class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[8, 16, 1], class_num=8):
        super(FrameWiseHGNN, self).__init__()
        self.activation = nn.Softmax(dim=-1)
        # hgnn layers
        self.layers = nn.ModuleList()  # the list of hypergraph layers
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(hids[j - 1], hids[j]))  # set hiddens
        self.cls_layer = nn.Linear(hids[-1], class_num)  # 设置全链接层，输入格式是1*8
        print("cls_layer:", self.cls_layer)

    def forward(self, x):
        print("x:", x.shape)
        layers = self.layers
        G = gen_G(x)
        print("G:", G.shape)
        f_in = torch.Tensor(x)
        # f_in = np.expand_dims(f_in, axis=1)
        # f_in = torch.Tensor(f_in)
        print("f_in:", f_in.shape)
        for L in layers:
            f_in = L(f_in, G)  # 进入网络层
            print("f:", f_in.size())
        # n x c -> 1 x c
        f_in = f_in.max(1)[0]
        print("f_in.max(1)", len(f_in))
        f_in = f_in.reshape((1, 18 * 3))
        # f_in = torch.from_numpy(f_in)
        # print("f_in.",f_in)
        # 1 x c -> 1 x 8
        out_feature = self.cls_layer(f_in)
        # print("out_feature:", out_feature)
        result = self.activation(out_feature)
        # result = softmax(out_feature)
        # print("result:", result)
        # result_max = result.argmax(dim=1)
        # print(result_max)

        return result


def train_it():
    train_dataset = MyDataSet(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    valid_dataset = MyDataSet(flag='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=10, shuffle=True)

    train_loss = []
    valid_loss = []
    train_epochs_loss = []
    valid_epochs_loss = []
    all_cnt = 0
    net = FrameWiseHGNN()
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        net.train()
        train_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
            # print(data_x)
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            optimizer.zero_grad()
            outputs = net(data_x)
            train_pred = torch.argmax(outputs)
            loss = criterion(train_pred, data_y)
            cnt_cor = (train_pred == data_y).sum()
            all_cnt += cnt_cor.item()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of train, loss={}, train acc is:{:6f}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item(), all_cnt/len(train_dataset)))
        train_epochs_loss.append(np.average(train_epoch_loss))


        # =====================valid============================
        net.eval()
        valid_epoch_loss = []
        all_val_cnt = 0
        for idx, (data_x, data_y) in enumerate(valid_dataloader, 0):
            data_x = data_x.to(device)
            data_y = data_y.to(device)
            outputs = net(data_x)
            valid_pred = torch.argmax(outputs)
            loss = criterion(valid_pred, data_y)
            cnt_val_cor = (valid_pred == data_y).sum()
            all_val_cnt += cnt_val_cor.item()
            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            if idx % (len(valid_dataloader) // 2) == 0:
                print("epoch={}/{},{}/{}of valid, loss={}, valid acc is:{:6f}".format(
                    epoch, args.epochs, idx, len(valid_dataloader), loss.item(), all_val_cnt/len(valid_dataset)))
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
    # train_it()
    # gen_G(get_train())
    get_valid()
    # FrameWiseHGNN(hids=[1, 36, 16], class_num= 8)
