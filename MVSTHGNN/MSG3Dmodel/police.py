import sys
sys.path.insert(0, '')
sys.path.extend(['../'])

import numpy as np

from .tools import get_adjacency_matrix, normalize_adjacency_matrix

num_node = 18
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 15), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
                    (8, 2), (9, 10), (10, 11), (11, 9), (12, 13), (13, 14),
                    (14, 12), (15, 16), (16, 17), (17, 18), (18, 1)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class AdjMatrixGraph:
    def __init__(self, *args, **kwargs):
        self.edges = neighbor
        self.num_nodes = num_node
        self.self_loops = [(i, i) for i in range(self.num_nodes)]
        self.A_binary = get_adjacency_matrix(self.edges, self.num_nodes)
        self.A_binary_with_I = get_adjacency_matrix(self.edges + self.self_loops, self.num_nodes)
        self.A = normalize_adjacency_matrix(self.A_binary)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    graph = AdjMatrixGraph()
    A, A_binary, A_binary_with_I = graph.A, graph.A_binary, graph.A_binary_with_I
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(A_binary_with_I, cmap='gray')
    ax[1].imshow(A_binary, cmap='gray')
    ax[2].imshow(A, cmap='gray')
    plt.show()
    print(A_binary_with_I.shape, A_binary.shape, A.shape)
