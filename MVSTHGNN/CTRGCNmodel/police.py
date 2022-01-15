import sys
import numpy as np

sys.path.extend(['../'])
from . import tools

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 15), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                    (8, 2), (9, 10), (10, 11), (11, 9), (12, 13), (13, 14),
                    (14, 12), (15, 16), (16, 17), (17, 18), (18, 1)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
