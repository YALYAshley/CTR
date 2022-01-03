# !/usr/bin/env python
import numpy as np
# 邻接矩阵
'''
 
对于无向图顶点之间存在边,则为1,反之则为0
 
   e1  e2  e3
v1 1   0   0
v2 0   1   0
v3 0   0   1
v4 1   0   0
v5 0   1   0
v6 0   0   1
v7 1   0   0
v8 0   1   0
v9 0   0   1
 
'''
 
# import numpy as np

# from pygraph.classes.hypergraph import hypergraph
# from pygraph.readwrite.dot import write_hypergraph

# h = hypergraph()

# h.add_nodes(['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7'])
# h.add_hyperedges(['e1', 'e2', 'e3', 'e4'])

# h.link('v1', 'e1')
# h.link('v2', 'e1')
# h.link('v3', 'e1')
# h.link('v2', 'e2')
# h.link('v3', 'e2')
# h.link('v3', 'e3')
# h.link('v5', 'e3')
# h.link('v6', 'e3')
# h.link('v4', 'e4')

# with open('hypergraph.dot', 'w') as f:
#     f.write(write_hypergraph(h))

# import sys
# import os
# from dataloader import MyDataset, DataLoader
# from json2data import json2data
# class Logger(object):
#   def __init__(self, filename="Default.log"):
#     self.terminal = sys.stdout
#     self.log = open(filename, "a")
#   def write(self, message):
#     self.terminal.write(message)
#     self.log.write(message)
#   def flush(self):
#     pass
# path = os.path.abspath(os.path.dirname(__file__))
# type = sys.getfilesystemencoding()
# sys.stdout = Logger('a.txt')
# data = json2data()

# #请先安装sklearn、numpy库
# from sklearn.metrics import precision_score, recall_score, f1_score
# import numpy as np
 
# y_true = np.array([[1, 1, 1],
#                    [1, 1, 0]])
# y_pred = np.array([[1, 0, 1],
#                    [1, 1, 1]])
 
# y_true = np.reshape(y_true, [-1])
# y_pred = np.reshape(y_pred, [-1])
 
# p = precision_score(y_true, y_pred, average='binary')
# r = recall_score(y_true, y_pred, average='binary')
# f1score = f1_score(y_true, y_pred, average='binary')
 
# print("train_precision:",p)
# print("train_recall:",r)
# print("train_f1score:",f1score)

from json2data import json2data

def get_packed_data(multi_data_sum, frame_st, frame_end):
   # packed_ft = None
   packed_multi_data = []
   interval_frame = (frame_end - frame_st) * 5
   _ , len_pose_file, len_views, len_lbls = json2data()

   for num in range(len_views):
      multi_data = []
      for i in range(0, len(multi_data_sum), len_pose_file):
         data2 = multi_data_sum[i + 10 * num: i + 10 * num + interval_frame]
         multi_data.append(data2)
      packed_multi_data.append(multi_data)
   # print(packed_multi_data)
   # print(len(multi_data_sum[2]))
   # print(len(packed_multi_data))

   packed_ft = []
   lbl_sum = []
   duration = 2

   for n in range(len(packed_multi_data)):
      f = []
      lbl = []
      for i in range(len_lbls * len_views):
         for j in range(0, interval_frame):
            f1 = packed_multi_data[n][i][j]['data']

            lbl1 = packed_multi_data[n][i][j]['action']
         f.append(f1)
         lbl.append(lbl1)
      # print(lbl)
      packed_ft.append(f1)
      lbl_sum.append(lbl)
      # print(f_sum)

   f_sum = []
   for end_frame_idx in range(interval_frame, len(packed_multi_data[0]), duration):
      cur_f, cur_lbl = get_packed_data(packed_multi_data, end_frame_idx - interval_frame, end_frame_idx)
      f_sum.append(cur_f)
      lbl_sum.append(cur_lbl)
   # print("f.size:",len(f))
   # print("lbl.size:",lbl_sum)
   return f_sum, lbl_sum


if __name__ == '__main__':

    multi_data_sum, len_pose_file, len_views, len_lbls = json2data()
    get_packed_data(multi_data_sum, 0, 1)