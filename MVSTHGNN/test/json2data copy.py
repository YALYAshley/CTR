import numpy as np
import torch
import os.path as osp
import json, os

def json2data():
    # body_id_dict = {'head': [0, 1, 14, 15, 16, 17],
    #                 'lefthand': [2, 3, 4],
    #                 'righthand':[5, 6, 7],
    #                 'leg': [8, 9, 10, 11, 12, 13]}
    body_id = ['head','head','lefthand','lefthand','lefthand',
                'righthand','righthand','righthand','leg',
                'leg','leg','leg','leg','leg','head','head','head','head']

    root = '/home/mn/8T/code/MVSTHGNN/data/json/'
    lbls = ['changelane','leftturnwait','pullover','slowdown','stop','straight','turnleft','turnright']
    lbls_trans = [0,1,2,3,4,5,6,7]
    # lbls_set = set(lbls)
    # #构建一个编号与名称对应的字典，以后输出的数字要变成动作名称的时候用：
    # lbls_list = list(lbls_set)
    # dic = {}
    # for i in range(8):
    #     dic[lbls_list[i]] = i
    # print(dic)
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
    # print(len(map))
    return map, len(pose_file), len(views), len(lbls)


if __name__ == '__main__':
    json2data()