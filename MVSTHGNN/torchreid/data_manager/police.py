from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os.path as osp
import numpy as np
import json

from torchreid.utils.iotools import read_json


class police(object):
    """
    police
    
    Dataset statistics:
    # actions: 8
    # 
    # cameras: 3
    """
    dataset_dir = 'police'

    def __init__(self, root='data', split_id=0, min_seq_len=0, verbose=True, **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.split_path = osp.join(self.dataset_dir, 'splits.json')
        self.cam_a_path = osp.join(self.dataset_dir, 'police', 'left')
        self.cam_b_path = osp.join(self.dataset_dir, 'police', 'center')
        self.cam_c_path = osp.join(self.dataset_dir, 'police', 'right')
        self.pose_file = osp.join(self.dataset_dir, 'poses.json')
        # print("left:",self.cam_a_path)
        # print("center:",self.cam_b_path)
        # print("right:",self.cam_c_path)

        self._check_before_run()
        with open(self.pose_file, 'r') as f:
            self.poses = json.load(f)
        # process the pose information
        self.process_poses = dict()
        # print("process_poses:",self.process_poses)
        for key in self.poses:
            # save only one body
            maxidx = -1
            maxarea = -1
            maxscore = -1
            # print("self.poses[key]['bodies']:",self.poses[key]['bodies'])
            assert len(self.poses[key]['bodies']) >= 1, 'pose of {} is empty'.format(key)
            if len(self.poses[key]['bodies']) == 1:
                self.process_poses[key] = np.array(self.poses[key]['bodies'][0]['joints']).reshape((-1, 3))
                # print("self.poses[key]['bodies'][0]['joints']",self.poses[key]['bodies'][0]['joints'])
            else:
                for idx, body in enumerate(self.poses[key]['bodies']):
                    tmp_kps = np.array(body['joints']).reshape((-1, 3))
                    tmp_area = (max(tmp_kps[:, 0]) - min(tmp_kps[:, 0])) * (max(tmp_kps[:, 1]) - min(tmp_kps[:, 1]))
                    tmp_score = body['score']
                    if tmp_score > maxscore:
                        if tmp_area > maxarea and tmp_score > 1.1 * maxscore:
                            maxscore = tmp_score
                            maxidx = idx
                self.process_poses[key] = np.array(self.poses[key]['bodies'][maxidx]['joints']).reshape((-1, 3))
        splits = read_json(self.split_path)
        if split_id >= len(splits):
            raise ValueError("split_id exceeds range, received {}, but expected between 0 and {}".format(split_id, len(splits)-1))
        split = splits[split_id]
        train_dirs, test_dirs = split['train'], split['test']
        print("# train identites: {}, # test identites {}".format(len(train_dirs), len(test_dirs)))
        print("train_dirs:",train_dirs)
        print("test_dirs:",test_dirs)

        train, num_train_tracklets, num_train_pids, num_imgs_train = \
          self._process_data(train_dirs, cam1=True, cam2=True, cam3=True)
        query, num_query_tracklets, num_query_pids, num_imgs_query = \
          self._process_data(test_dirs, cam1=True, cam2=False, cam3=False)
        gallery, num_gallery_tracklets, num_gallery_pids, num_imgs_gallery = \
          self._process_data(test_dirs, cam1=False, cam2=True, cam3=False)

        num_imgs_per_tracklet = num_imgs_train + num_imgs_query + num_imgs_gallery
        min_num = np.min(num_imgs_per_tracklet)
        max_num = np.max(num_imgs_per_tracklet)
        avg_num = np.mean(num_imgs_per_tracklet)

        num_total_pids = num_train_pids + num_query_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        if verbose:
            print("=> police loaded")
            print("Dataset statistics:")
            print("  ------------------------------")
            print("  subset   | # ids | # tracklets")
            print("  ------------------------------")
            print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_tracklets))
            print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_tracklets))
            print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets))
            print("  ------------------------------")
            print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_tracklets))
            print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
            print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))

    def _process_data(self, dirnames, cam1=True, cam2=True, cam3=True):
        tracklets = []
        num_imgs_per_tracklet = []
        dirname2pid = {dirname:i for i, dirname in enumerate(dirnames)}
        
        for dirname in dirnames:
            if cam1:
                # person_dir = osp.join(self.cam_a_path, dirname)
                # print("person_dir1:",person_dir)
                # img_names = sorted(glob.glob(osp.join(person_dir, '*.jpg')))
                img_names = osp.join(self.cam_a_path, dirname)
                print("ima_names1:",img_names)
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 0))
                num_imgs_per_tracklet.append(len(img_names))

            if cam2:
                # person_dir = osp.join(self.cam_b_path, dirname)
                # img_names = sorted(glob.glob(osp.join(person_dir, '*.jpg')))
                img_names = osp.join(self.cam_b_path, dirname)
                print("ima_names2:",img_names)
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 1))
                num_imgs_per_tracklet.append(len(img_names))

            if cam3:
                # person_dir = osp.join(self.cam_c_path, dirname)
                # img_names = sorted(glob.glob(osp.join(person_dir, '*.jpg')))
                img_names = osp.join(self.cam_c_path, dirname)
                print("ima_names3:",img_names)
                assert len(img_names) > 0
                img_names = tuple(img_names)
                pid = dirname2pid[dirname]
                tracklets.append((img_names, pid, 2))
                num_imgs_per_tracklet.append(len(img_names))

        num_tracklets = len(tracklets)
        num_pids = len(dirnames)

        return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet
