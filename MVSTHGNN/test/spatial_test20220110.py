from __future__ import absolute_import
from __future__ import division
from HGNN import HGNN_conv
from HGNN import generate_G_from_H, hyperedge_concat
from json2data import json2data
from sklearn.metrics import precision_score, recall_score,f1_score

# __all__ = ['MV_TSHL']

import numpy as np
import torch
from torch import nn
import math
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from construct_hypergraph import construct_st_H, construct_spatial_H, construct_temporal_H

device = "cpu"

class argparse():
    pass


args = argparse()
args.epochs, args.learning_rate, args.patience = [100, 0.01, 4]
args.device, = [torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), ]
train_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/train/json/'
valid_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/valid/json/'
multi_train_data, train_lbls = json2data(train_root)
multi_valid_data, valid_lbls = json2data(valid_root)
import warnings
warnings.filterwarnings("ignore")

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
            frame_end = 1
            duration = (frame_end - frame_st) * 5
            self.data, self.label = get_packed_data(multi_train_data, train_lbls, frame_st, frame_end)
            for i in range(duration):
                self.data.pop()
                self.label.pop()
        else:
            frame_st = 0
            frame_end = 1
            duration = (frame_end - frame_st) * 5
            self.data, self.label = get_packed_data(multi_valid_data, valid_lbls, frame_st, frame_end)
            for i in range(duration):
                self.data.pop()
                self.label.pop()

        self.length = len(self.data)


    def __getitem__(self, index):

        data = self.data[index]
        label = self.label[index]
        # n_frame * 3_view * 18 * xyp ---->> n * c
        data = data.view(-1, 3)
        # data = data - data.min(0)[0]
        data = (data - data.min(0)[0])/(data.max(0)[0] - data.min(0)[0])

        # construct G

        H = None
        cur_x = data
        # gen G: n * n
        H_st = construct_st_H(cur_x)
        H = hyperedge_concat(H, H_st)
        H_spatial = construct_spatial_H(cur_x)
        H = hyperedge_concat(H, H_spatial)
        H_temporal = construct_temporal_H(cur_x)
        H = hyperedge_concat(H, H_temporal)
        self.G = generate_G_from_H(H)
        # end G

        return data, label, self.G

    def __len__(self):
        return self.length

class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[3, 32, 128, 512, 128], class_num=8):
        super(FrameWiseHGNN, self).__init__()
        self.activation = nn.Softmax(dim=1)
        # hgnn layers
        self.layers = nn.ModuleList()  # the list of hypergraph layers
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(hids[j - 1], hids[j]))  # set hiddens
        self.cls_layer = nn.Linear(hids[-1], class_num)  # ????????????????????????????????????1*8
        print("cls_layer:", self.cls_layer)

    def forward(self, x, G):

        for L in self.layers:
            x = L(x, G)  # ???????????????
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training)
        # bz * n * c -> bz * c
        x = x.max(1)[0]
        # 1 x c -> 1 x 8
        out_feature = self.cls_layer(x)

        # result = self.activation(out_feature)
        result = out_feature
        return result

def train_it():
    train_dataset = MyDataSet(flag='train')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    valid_dataset = MyDataSet(flag='valid')
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True)

    train_loss = []
    train_epochs_loss = []

    net = FrameWiseHGNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        net.train()
        train_epoch_loss = []
        train_pred_list = []
        lbl_list = []
        for idx, (data_x, data_y, G) in enumerate(train_dataloader, 0):
            data = data_x.to(torch.float32).to(device)
            lbl = data_y.long().to(device)   #torch.int64
            lbl = lbl.view(lbl.size(0))

            G = train_dataloader.dataset.G
            G = G.unsqueeze(0)
            Gs = G.repeat (data.size(0), 1, 1)

            optimizer.zero_grad()

            # compute output
            outputs = net(data, Gs)  #torch.float32
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

        print(f"epoch={epoch}/{args.epochs}, loss = {loss.item():.4f}, training: acc {acc * 100:.3f}%, "
              f"precision {pre_score * 100:.3f}%, recall {rec_score:.3f}, f1 {f_score:.3f}")

        train_epochs_loss.append(np.average(train_epoch_loss))

        # =====================valid============================
        net.eval()
        valid_pred_list = []
        lbl_valid_list = []
        with torch.no_grad():
            for idx, (data_x, data_y, G) in enumerate(valid_dataloader, 0):
                data_valid = data_x.to(torch.float32).to(device)
                lbl_valid = data_y.long().to(device)  # torch.int64
                lbl_valid = lbl_valid.view(lbl_valid.size(0))

                # compute output
                G = valid_dataloader.dataset.G
                G = G.unsqueeze(0)
                Gs = G.repeat(data_valid.size(0), 1, 1)
                outputs = net(data_valid, Gs)

                valid_pred = outputs.argmax(dim=1)
                valid_pred_list.extend(valid_pred.cpu().numpy().tolist())
                lbl_valid_list.extend(lbl_valid.cpu().numpy().tolist())

        valid_pred_result = np.array(valid_pred_list)
        lbl_valid_result = np.array(lbl_valid_list)
        valid_acc = np.mean(valid_pred_result == lbl_valid_result)
        pre_valid_score = precision_score(lbl_valid_result, valid_pred_result, average='weighted')
        rec_valid_score = recall_score(lbl_valid_result, valid_pred_result, average='weighted')
        f_valid_score = f1_score(lbl_valid_result, valid_pred_result, average='weighted')

        print(f"epoch={epoch}/{args.epochs}, validing: acc {valid_acc * 100:.3f}%, "
              f"precision {pre_valid_score * 100:.3f}%, recall {rec_valid_score:.3f}, f1 {f_valid_score:.3f}")

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