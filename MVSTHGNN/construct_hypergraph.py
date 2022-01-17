import torch
import numpy as np
import scipy.linalg


def construct_st_H(data):
    """

    Args:
        data: [n_frame, 3_view, 18joints], 3

    Returns:
        H: [18 * 3 * n_frames], 18 * n_frame np,array

    """
    views = 3
    joints = 18
    n_frame = int(len(data) / (views * joints))
    n_obj = views * joints * n_frame
    n_edge = joints
    H = np.ones((len(data), n_edge * n_frame)) * 0.0001
    for edg_idx in range(n_edge):
        for node_idx in range(edg_idx, len(data), joints):
            H[node_idx, edg_idx] = 1.0
    return H


def construct_spatial_H(data):
    """

    Args:
        data: [n_frame, 3_view, 18joints], 3

    Returns:
        H: [18 * 3 * n_frames], 18 * n_frame

    """
    # 三个视角的对应点连到一个超边
    views = 3
    joints = 18
    n_frame = int(len(data) / (views * joints))
    n_obj = joints
    n_edge = 3
    H = np.zeros((n_obj, n_edge))
    ori_index = [(0, 1, 14, 15, 16, 17), (2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13)]

    for edg_idx in range(n_edge):
        for j in range(6):
            node_idx = ori_index[edg_idx][j]
            H[node_idx, edg_idx] = 1.0
    #
    big_H = np.zeros((joints * n_frame, n_edge * n_frame))
    for i in range(n_frame):
        bias_x = i * joints
        bias_y = i * n_edge
        big_H[bias_x : bias_x + joints, bias_y : bias_y + n_edge] = H
    return big_H


def construct_temporal_H(data):
    """

    Args:
        data: [n_frame, 3_view, 18joints], 3

    Returns:
        H: [18 * 3 * n_frames], 18 * 3_view

    """
    views = 3
    joints = 18
    n_frame = int(len(data) / (views * joints))
    n_obj = len(data)
    n_edge = joints
    H = np.ones((n_obj, n_edge * n_frame)) * 0.0001
    for edg_idx in range(n_edge * views):
        for node_idx in range(edg_idx, len(data), joints * views):
            H[node_idx, edg_idx] = 1.0
    return H
