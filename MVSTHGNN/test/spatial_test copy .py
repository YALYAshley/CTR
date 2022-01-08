from __future__ import absolute_import
from __future__ import division
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H, hyperedge_concat
from json2data import json2data
from torchreid.metrics import metric
from torchreid.utils.avgmeter import AverageMeter

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
train_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/train/json/'
valid_root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/valid/json/'
multi_train_data, train_lbls = json2data(train_root)
multi_valid_data, valid_lbls = json2data(valid_root)

def get_packed_data(multi_data_sum, multi_lbls, frame_st, frame_end):

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


def get_feature(data,label):

    x = []
    y = []
    for i in range(len(data)):
        for j in range(len(data[0])):  # 5
            f1 = []
            for joint in range(len(data[0][0][0])):  # 18 joints
                f = []
                for view in range(len(data[0][0])):  # 3 views
                    X = data[i][j][view][joint]
                    f.append(X)
                f1.append(f)
            x.append(f1)
            lbl1 = label[i][j]
            y.append(lbl1)
    return x, y

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

def gen_G(X):
    G = []

    for feature in X:  # iter over each batch
        for ft in feature:
            H = None
            ft = tuple(ft)
            ft = torch.cat(ft, dim=0)
            ft = ft.reshape(-1, 3)
            H_spatial = construct_H_with_KNN(ft, K_neigs=3,
                                             is_probH=True,
                                             m_prob=1, mode="global", total_node_num=len(feature), S=3)
            H = hyperedge_concat(H, H_spatial)
            g = generate_G_from_H(H)
        G.append(g)
    G = torch.stack((G), dim=0)
    return G

class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[3, 1], class_num=8):
        super(FrameWiseHGNN, self).__init__()
        self.activation = nn.Softmax(dim=1)
        # hgnn layers
        self.layers = nn.ModuleList()  # the list of hypergraph layers
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(hids[j - 1], hids[j]))  # set hiddens
        self.cls_layer = nn.Linear(hids[-1], class_num)  # 设置全链接层，输入格式是1*8
        print("cls_layer:", self.cls_layer)

    def forward(self, x):
        layers = self.layers
        f_in = gen_G(x)
        for L in layers:
            f_in = L(f_in, f_in)  # 进入网络层
        # n x c -> 1 x c
        f_in = f_in.max(1)[0]
        # 1 x c -> 1 x 8
        out_feature = self.cls_layer(f_in)

        # result = self.activation(out_feature)
        result = out_feature
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

    net = FrameWiseHGNN()
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    for epoch in range(args.epochs):
        net.train()
        train_epoch_loss = []

        for idx, (data_x, data_y) in enumerate(train_dataloader, 0):
            data = data_x.to(torch.float32).to(device)
            lbl = data_y.to(torch.float32).to(device)
            optimizer.zero_grad()

            data_x, data_y = get_feature(data, lbl)
            data_y = torch.Tensor(data_y).long()
            data_y_acc = np.array(data_y)

            # compute output
            outputs = net(data_x)
            loss = criterion(outputs, data_y)

            #measure accuracy and record loss
            train_pred = outputs.argmax(dim=1)
            train_pred = np.array(train_pred)
            confMatrix,label2idx = metric.buildConfusionMatrix(train_pred,data_y_acc)

            # all_prediction = metric.calculate_all_prediction(confMatrix)
            label_prediction = []
            label_recall = []

            for i in label2idx:
                label_prediction.append(metric.calculate_label_prediction(confMatrix,label2idx[i]))
                label_recall.append(metric.calculate_label_recall(confMatrix,label2idx[i]))
                # for j in label2idx:
                #     labelidx_i = label2idx[i]
                #     label2idx_j = label2idx[j]
                #     print('  ',confMatrix[labelidx_i][label2idx_j],end=' ')
                # print('\n')
        
            # print('prediction(accuracy)=',all_prediction,'%')
            # for ei,i in enumerate(label2idx):
            #     print(ei,'\t',i,'\t','prediction=',label_prediction[ei],'%,\t recall=',label_recall[ei],'%,\tf1=',
            #           metric.calculate_f1(label_prediction[ei],label_recall[ei]))
            p = round(np.array(label_prediction).sum()/len(label_prediction),2)
            r = round(np.array(label_recall).sum()/len(label_prediction),2)

            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            train_loss.append(loss.item())
            if idx % (len(train_dataloader) // 2) == 0:
                print("train: epoch={}/{},{}/{}of train, loss={}".format(
                    epoch, args.epochs, idx, len(train_dataloader), loss.item()))
                print('prediction=', p, '%,recall=', r, '%,f1=', metric.calculate_f1(p, r))
        train_epochs_loss.append(np.average(train_epoch_loss))


        # =====================valid============================
        net.eval()
        valid_epoch_loss = []
        for idx, (data_x, data_y) in enumerate(valid_dataloader, 0):
            data = data_x.to(torch.float32).to(device)
            lbl = data_y.to(torch.int32).to(device)

            data_x, data_y = get_feature(data, lbl)
            data_y = torch.Tensor(data_y).long()
            data_y_acc = np.array(data_y)

            outputs = net(data_x)
            loss = criterion(outputs, data_y)

            # measure accuracy and record loss
            valid_pred = outputs.argmax(dim=1)
            valid_pred = np.array(valid_pred)
            confMatrix, label2idx = metric.buildConfusionMatrix(valid_pred, data_y_acc)

            label_prediction = []
            label_recall = []

            for i in label2idx:
                label_prediction.append(metric.calculate_label_prediction(confMatrix, label2idx[i]))
                label_recall.append(metric.calculate_label_recall(confMatrix, label2idx[i]))

            p = round(np.array(label_prediction).sum() / len(label_prediction), 2)
            r = round(np.array(label_recall).sum() / len(label_prediction), 2)


            valid_epoch_loss.append(loss.item())
            valid_loss.append(loss.item())
            if idx % (len(valid_dataloader) // 2) == 0:
                print("valid: epoch={}/{},{}/{}of valid, loss={}".format(
                    epoch, args.epochs, idx, len(valid_dataloader), loss.item()))
                print('prediction=', p, '%,recall=', r, '%,f1=', metric.calculate_f1(p, r))
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
