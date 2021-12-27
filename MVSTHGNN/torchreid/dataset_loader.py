from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import os.path as osp
import random
import platform

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchreid.transforms import ImageData


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, pid, camid


class VideoDataset(Dataset):
    """Video Person ReID Dataset.
    Note batch data has shape (batch, seq_len, channel, height, width).
    """
    sample_methods = ['evenly', 'random', 'all', 'consecutive', 'dense', 'restricted', 'skipdense']

    def __init__(self, dataset, seq_len=15, sample='evenly', transform=None, training=False, pose_info=None,
                 num_split=8, num_parts=3, num_scale=True, pyramid_part=True, enable_pose=True,mode='all_points',
                 node_num=18,crop_scale_h=0.5, crop_scale_w=0.5,max_len=1000):
        self.dataset = dataset
        self.seq_len = seq_len
        self.sample = sample
        self.transform = transform
        self.training = training
        self.pose_info = pose_info
        self.num_split = num_split
        self.num_parts = num_parts
        self.num_scale = num_scale
        self.pyramid_part = pyramid_part
        self.enable_pose = enable_pose
        self.crop_scale_h = crop_scale_h
        self.crop_scale_w = crop_scale_w
        self.mode = mode
        if self.mode == 'all_points':
            self.node_num = pose_info[random.choice(list(pose_info))].shape[0]  # the node num in the hypergraph
        elif self.mode in ['part_points','multi_graph','all_graph']:
            self.node_num = node_num
        # XXX: make sure max_len not too big and divided by seq_len
        self.max_len = max_len

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        print('/n')
        print('*******************************************\n')
        print('self.dataset: ', self.dataset)
        print('*******************************************\n')
        print('/n')
        img_paths, pid, camid = self.dataset[index]
        num = len(img_paths)
        print("@@@@@@@@@@@@@@@@@__num",num)
                # abandon the over length images
        if num > self.max_len:
            num = self.max_len
            img_paths = img_paths[:num]

        if self.sample == 'random':
            """
            Randomly sample seq_len items from num items,
            if num is smaller than seq_len, then replicate items
            """
            indices = np.arange(num)
            replace = False if num >= self.seq_len else True
            indices = np.random.choice(indices, size=self.seq_len, replace=replace)
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)
        elif self.sample == 'evenly':
            """
            Evenly sample seq_len items from num items.
            """
            if num >= self.seq_len:
                num -= num % self.seq_len
                indices = np.arange(0, num, num / self.seq_len)
            else:
                # if num is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num)
                num_pads = self.seq_len - num
                indices = np.concatenate([indices, np.ones(num_pads).astype(np.int32) * (num - 1)])
            assert len(indices) == self.seq_len
        elif self.sample == 'all':
            """
            Sample all items, seq_len is useless now and batch_size needs
            to be set to 1.
            """
            indices = np.arange(num)
        elif self.sample == 'consecutive':
            """
            Randomly sample seq_len consecutive frames from num frames,
            if num is smaller than seq_len, then replicate items.
            This sampling strategy is used in training phase.
            """
            frame_indices = np.arange(num)
            rand_end = max(0, num - self.seq_len - 1)
            begin_index = random.randint(0, rand_end)
            end_index = min(begin_index + self.seq_len, num)

            indices = frame_indices[begin_index:end_index]

            for index in indices:
                if len(indices) >= self.seq_len: break
                np.append(indices, index)
        elif self.sample == 'dense':
            """
            Sample all frames in a video into a list of clips, each clip contains seq_len frames, 
            batch_size needs to be set to 1. This sampling strategy is used in test phase.
            """
            indices = np.arange(num)
            append_size = self.seq_len - num % self.seq_len
            indices = np.append(indices, [num - 1] * append_size)
        elif self.sample == 'restricted':
            total_indices = np.arange(num)
            append_size = self.seq_len - num % self.seq_len
            total_indices = np.append(total_indices, [num - 1] * append_size)

            chunk_size = len(total_indices) // self.seq_len
            indices = []
            for seq_idx in range(self.seq_len):
                chunk_index = total_indices[seq_idx * chunk_size: (seq_idx + 1) * chunk_size]
                # idx = np.random.choice(chunk_index, 1)
                print("chunk_index",chunk_index)
                fail_times = 0
                while True:  # only choose the image with existing pose
                    idx = np.random.choice(chunk_index, 1)
                    print("idx",idx)
                    path = img_paths[idx[0]]
                    #path = "/home/mn/桌面/STHGNN/data/police/left/left_1276.jpg"
                    print('*******************************************\n')
                    print("@@@@@@@@@@@@@@@@@@@@@",img_paths)
                    print("@@@@@@@@@@@@@@@@@@@@",idx[0])
                    print('111111111111111111111',path)
                    print('*******************************************\n')
                    key = get_key(path)
                    if key in self.pose_info:  # have pose information
                        indices.append(idx)
                        break
                    else:
                        fail_times += 1
                        if fail_times >= 50:  # break and change to random choose
                            break
                # random choose one image with pose information
                if fail_times >= 50:
                    random_fail_times = 0
                    while True:  # tries to choose the image with existing pose
                        idx = np.random.choice(total_indices, 1)
                        path = img_paths[idx[0]]
                        key = get_key(path)
                        if key in self.pose_info:  # have pose information
                            indices.append(idx)
                            break
                        else:
                            random_fail_times += 1
                            if random_fail_times >= 150:  # break, fail to catch an image with pose information
                                indices.append(idx)
                                break
            indices = np.sort(indices)
        elif self.sample == 'skipdense':
            """
            Sample all frames in the video into a list of frames, and frame index is increased by video_len / seq_len.
            """
            indices = np.arange(num)
            append_size = self.seq_len - num % self.seq_len
            indices = np.append(indices, [num - 1] * append_size)
            skip_len = len(indices) // self.seq_len
            final_indices = []
            for i in range(skip_len):
                final_indices.extend([indices[idx] for idx in np.arange(i, len(indices), skip_len)])
            indices = final_indices
        else:
            raise KeyError("Unknown sample method: {}. Expected one of {}".format(self.sample, self.sample_methods))

        imgs = []
        pose_loc = [] # save the cropped part according to pose location (left, top)
        img_sizes = []
        for index in indices:
            # images
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            img_sizes.append(img.size)
            imgs.append(ImageData(img))
            # pose location
            assert isinstance(self.pose_info, dict), 'please load the pose info'
            locs = get_crop_loc(path=img_path, crop_scale_h=self.crop_scale_h,crop_scale_w=self.crop_scale_w,
                                    poses=self.pose_info, node_num=self.node_num, mode=self.mode)
            pose_loc.append(locs)

        if self.transform is not None:
            imgs = self.transform(imgs)
            imgs = [img.img for img in imgs]

            imgs = torch.stack(imgs, dim=0)
            pose_loc = torch.Tensor(pose_loc)

        return imgs, pid, camid, pose_loc


def get_crop_loc(path,crop_scale_h,crop_scale_w,poses,node_num=18,mode='all_points'):
    """
    get the cropped location from pose information
    :return: sub_locs: ndarray list of cropped location, in ratio
    """
    sub_locs = []
    im = read_image(path)
    key = get_key(path)

    crop_sz_x = int(crop_scale_h * im.size[0])
    crop_sz_y = int(crop_scale_w * im.size[1])
    if key in poses: # have pose information
        temp_pose = poses[key]
        # if mode == 'all_points':
        for idx, (x, y, confi) in enumerate(temp_pose):
            x, y = int(x), int(y)
            left, right = max(x - int(crop_sz_x / 2),0) / im.size[0], min(x + int(crop_sz_x / 2),im.size[0]-1) / im.size[0]
            top, bottom = max(y - int(crop_sz_y / 2),0) / im.size[1], min(y + int(crop_sz_y / 2),im.size[1]-1) / im.size[1]
            sub_locs.append([left, top])
    else:
        # random crop locations
        x_indices = np.arange(int(crop_sz_x / 2), im.size[0]-int(crop_sz_x / 2))
        y_indices = np.arange(int(crop_sz_y / 2), im.size[1]-int(crop_sz_y / 2))
        if mode == 'all_points':
            total_node_num = 18
        elif mode in ['part_points', 'multi_graph','all_graph']:
            total_node_num = sum(node_num)
        for idx in range(total_node_num):
            x, y = np.random.choice(x_indices, 1)[0], np.random.choice(y_indices, 1)[0]
            left, right = max(x - int(crop_sz_x / 2), 0) / im.size[0], min(x + int(crop_sz_x / 2), im.size[0] - 1) / \
                          im.size[0]
            top, bottom = max(y - int(crop_sz_y / 2), 0) / im.size[1], min(y + int(crop_sz_y / 2), im.size[1] - 1) / \
                          im.size[1]
            sub_locs.append([left, top])
    return sub_locs

def get_key(path):
    """
    return the key of the given path
    :param path:
    :return: key
    """
    system = platform.system() # Windows or Linux
    if 'ilids-vid' in path:  # ilidsvid
        # key = path.split('/')[-1] # linux
        key = path.split('/')[-1].split('\\')[-1] # windows + linux
    elif 'prid2011' in path:  # prid2011
        key = '-'.join(path.split('/')[-3:]) if system == 'Linux' else '-'.join(path.split('\\')[-3:])
    elif 'mars' in path:  # mars
        key = path.split('/')[-1] if system == 'Linux' else path.split('\\')[-1]
    elif 'duke' in path:  # dukemtmcvidreid
        print("~~~~")
        key = '-'.join(path.split('/')[-3:]) if system == 'Linux' else '-'.join(path.split('\\')[-3:])
    elif 'police' in path:  # dukemtmcvidreid
        print("!!!!!!!!!!!!!!!!!!!!!!!  police !!!!!!!!!!!!")
        key = '-'.join(path.split('/')[-3:]) if system == 'Linux' else '-'.join(path.split('\\')[-3:])
        print("key",key)
    else:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`")
        raise ValueError('{} is not acceptable'.format(path))
    if len(key.split('/'))>1 or len(key.split('\\'))>1:
        raise ValueError('key of {} is in wrong format, check funtion get_key()'.format(path))
    return key

