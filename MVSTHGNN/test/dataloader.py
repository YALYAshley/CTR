import os
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch
import os.path as osp
import json, os
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self):
        
        action_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
        action_class.sort()
        class_indices = dict((k, v) for v, k in enumerate(action_class))
        lbls = action_class
        body_id = ['head','head','lefthand','lefthand','lefthand',
                'righthand','righthand','righthand','leg',
                'leg','leg','leg','leg','leg','head','head','head','head']
        for cla in action_class:
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            print(image_class)
        lbls_trans = image_class
        views = ['center', 'left', 'right']
        data = []

        map = []
        
        for lbl in lbls:
            for lbl_trans in lbls_trans:
                for view in views:
                    dataset_dir = osp.join(root, lbl, view)
                    pose_file = [pos_json for pos_json in os.listdir(dataset_dir) if pos_json.endswith('.json')]
                    pose_file.sort()
                    for length in range(0, len(pose_file)):
                        data_dir = osp.join(dataset_dir,pose_file[length])
                        with open(data_dir, 'r', encoding='utf-8-sig') as file:
                            poses = json.load(file)
                        for poses_key in poses:
                                data1 = []
                            # for i in range(0,len(poses[poses_key]['bodies'])):
                                for j in range(0,18):
                                    coor_x = np.array(poses[poses_key]['bodies'][0]['joints'][3*j])
                                    coor_y = np.array(poses[poses_key]['bodies'][0]['joints'][1+3*j])
                                    coor_prob = np.array(poses[poses_key]['bodies'][0]['joints'][2+3*j])
                                    data1.append(float(coor_x))
                                    data1.append(float(coor_y))
                                    data1.append(float(coor_prob))
                                data = torch.tensor(data1)
                                data = data.view(-1,18,3)
                                data_map = {'action': lbl_trans, 'view': view, 'data_file': pose_file[length],'data': data, 'key_map': body_id}
                                map.append(data_map)

    # The sliding window is 2s, data——>dict

        f1 = []
        f_sum = []
        lbl_sum = []
    
        multi_data_sum=[]
        windows = 2
        frames = 5
        for num in range(len(views)):
            multi_data = []
            for i in range(0, len(map), len(pose_file)):
                data2 = map[i + 10 * num: i + 10 * num + (windows * frames)]
                multi_data.append(data2)
            multi_data_sum.append(multi_data)

        loop_num = int(len(pose_file)/(windows * frames))
        for n in range(loop_num):
            f = []
            lbl = []
            for i in range(len(lbls)):
                for joint in range(0,18):
                    for j in range(0,(windows * frames)):
                        for view in range(0, len(views)):
                            f1 = multi_data_sum[n][view + 3 * i][j]['data'][0,joint,:]
                            f.append(f1)
                            lbl1 = multi_data_sum[n][view + 3 * i][j]['action']
                        lbl.append(lbl1)
                # print(lbl)
            f_sum.append(f)
            lbl_sum.append(lbl)
        self._x = f_sum
        self._y = lbls
        self._len = len(f_sum)

    def __len__(self):
        
        return self._len

    def __getitem__(self, idx):
        
        return self._x[idx],self._y[idx]
        

root = '/home/mn/8T/code/MVSTHGNN/data/json/'
data = MyDataset()
print(data)


