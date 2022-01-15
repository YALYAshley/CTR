import  torch
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
    n_frame = int(len(data)/(views * joints))
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
    views = 3
    joints = 18
    n_frame = int(len(data)/(views * joints))
    n_obj = views * joints
    n_edge = joints
    H = np.zeros((n_obj, n_edge))
    for edg_idx in range(n_edge):
        for node_idx in range(edg_idx, n_obj, joints):
            H[node_idx, edg_idx] = 1.0
    gen_H = scipy.linalg.block_diag(H,H,H,H,H)
    return gen_H

def construct_temporal_H(data):
    """

    Args:
        data: [n_frame, 3_view, 18joints], 3

    Returns:
        H: [18 * 3 * n_frames], 18 * 3_view

    """
    views = 3
    joints = 18
    n_frame = int(len(data)/(views * joints))
    n_obj = len(data)
    n_edge = joints
    H = np.ones((n_obj, n_edge * n_frame)) * 0.0001
    for edg_idx in range(n_edge * views):
        for node_idx in range(edg_idx, len(data), joints * views):
            H[node_idx, edg_idx] = 1.0
    return H

