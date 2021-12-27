import numpy as np
import torch
import os.path as osp
import json, os
import random

def json2data():
    
    random.seed(0)
    root = '/home/mn/8T/code/MVSTHGNN/data/json/'

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
    

    
    views = ['center', 'left', 'right']
    data = []

    map = []
    
    
    for lbl in lbls:
        # for lbl_trans in lbls_trans:
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
                            data_map = {'action': lbl, 'view': view, 'data_file': pose_file[length],'data': data, 'key_map': body_id}
                            map.append(data_map)   
    # print(map)
    return map, len(pose_file), len(views), len(lbls)


if __name__ == '__main__':
    json2data()