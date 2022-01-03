import numpy as np
import torch
import os.path as osp
import json, os
import random

def json2data(root):

    random.seed(0)
    # root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/json/'

    body_id = ['head','head','lefthand','lefthand','lefthand',
                'righthand','righthand','righthand','leg',
                'leg','leg','leg','leg','leg','head','head','head','head']

    action_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root,cla))]
    action_class.sort()
    class_indices = dict((k,v) for v, k in enumerate(action_class))
    # print(class_indices)
    json_str = json.dumps(dict((val,key)for key, val in class_indices.items()), indent = 4)
    with open('class_indices.json','w') as json_file:
        json_file.write(json_str)
    #lbls = ['changelane','leftturnwait','pullover','slowdown','stop','straight','turnleft','turnright']
    lbls = action_class
    lbls_trans = [0, 1, 2, 3, 4, 5, 6, 7]
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
                            data_map = {'lbl': lbl_trans, 'view': view, 'data_file': pose_file[length],'data': data, 'key_map': body_id}
                            map.append(data_map)   
    # print(len(map))
    # return map, len(pose_file), len(views), len(lbls)
    all_data = map
    len_pose_file= len(pose_file)
    len_views = len(views)
    len_lbls = len(lbls)
    packed = []
    lbl3 = []
    all_data1 = []
    num = 0

    for act in range(len_lbls) :
        for strat in range(len_pose_file):
            lbl1 = []
            for i in range(strat, len_pose_file * len_views, len_pose_file):
                data2 = all_data[i + act * 90]['data']
                lbl = all_data[i + act * 90]['lbl']
                num += 1
                all_data1.append(data2)
                lbl1.append(lbl)
                if num == len_views:
                    lbl2 = set(lbl1)
                    packed.append(all_data1)
                    num = 0
                    lbl3.append(lbl2)
    packed = tuple(all_data1)
    packed = torch.cat(packed, dim=0)
    packed_data = packed.reshape(-1, 3, 18, 3)
    packed_lbl = tuple(lbl3)

    return packed_data, packed_lbl


# if __name__ == '__main__':
#     root = '/home/mn/8T/code/new-hgnn/MVSTHGNN/data/train/json/'
#     json2data(root)
#     packed_data()