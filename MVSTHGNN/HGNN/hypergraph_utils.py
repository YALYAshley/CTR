import numpy as np
import torch

body_id_dict = {'head': [0, 1, 14, 15, 16, 17],
                'trunk': [2, 3, 4, 5, 6, 7],
                'leg': [8, 9, 10, 11, 12, 13]}

def Eu_dis(x):
    """
    Calculate the distance among each raw of x     计算x的每个原始点之间的距离
    :param x: N X D
                N: the object number
                D: Dimension of the feature 
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)    
    ab = x * x.T     # (3 * 3)
    dist_mat = aa + aa.T - 2 * ab  #(3 * 3)
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def feature_concat(*F_list, normal_col=False):
    """
    Concatenate multiple modality feature. If the dimension of a feature matrix is more than two,
    the function will reduce it into two dimension(using the last dimension as the feature dimension,
    the other dimension will be fused as the object dimension)
    :param F_list: Feature matrix list  
    :param normal_col: normalize each column of the feature  
    :return: Fused feature matrix 
    """
    features = None
    for f in F_list:
        if f is not None and f != []:
            # deal with the dimension that more than two
            if len(f.shape) > 2:
                f = f.reshape(-1, f.shape[-1])
            # normal each column
            if normal_col:
                f_max = np.max(np.abs(f), axis=0)
                f = f / f_max
            # facing the first feature matrix appended to fused feature matrix
            if features is None:
                features = f
            else:
                features = np.hstack((features, f))
    if normal_col:
        features_max = np.max(np.abs(features), axis=0)
        features = features / features_max
    return features


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list  
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix   
    :return: Fused hypergraph incidence matrix   
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix  
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    # print("H1:",H)
    return H


def generate_G_from_H(H, variable_weight=False, edge_weight=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight, edge_weight=edge_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight),edge_weight=edge_weight)
            print("G1:",G)
        return G


def _generate_G_from_H(H, variable_weight=False, edge_weight=None):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    # print("generate_H:",H.shape[1])
    n_edge = H.shape[0]
    # n_edge = 3

    if variable_weight:
        H = torch.Tensor(H)
        if edge_weight.is_cuda and not H.is_cuda:
            H = H.to('cuda')
        # the weight of the hyperedge
        W = edge_weight
        # the degree of the node
        DV = torch.sum(H * W, 1)
        # the degree of the hyperedge
        DE = torch.sum(H, 0)

        invDE = torch.diag(DE.pow(-1))
        DV2 = torch.diag(DV.pow(-0.5))
        W = torch.diag(W)
        HT = H.t()

        DV2_H = torch.mm(DV2, H)  #DV2和H相乘
        invDE_HT_DV2 = torch.mm(torch.mm(invDE, HT), DV2)
        G = torch.mm(torch.mm(DV2_H, W), invDE_HT_DV2)
        # print("G2:",G)
        return G
    else:
        H = torch.Tensor(H)

        W = torch.Tensor(np.ones(n_edge))
        
        if W.is_cuda and not H.is_cuda:
            H = H.to('cuda')
        # print(H.shape,W.size())
        DV = torch.sum(H * W, 1)
        DE = torch.sum(H, 0)

        invDE = torch.diag(DE.pow(-1))
        DV2 = torch.diag(DV.pow(-0.5))
        W = torch.diag(W)
        HT = H.t()

        DV2_H = torch.mm(DV2, H)
        invDE_HT_DV2 = torch.mm(torch.mm(invDE, HT), DV2)
        G = torch.mm(torch.mm(DV2_H, W), invDE_HT_DV2)
        # print("G3:",G)
        return G



def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix  由超图节点距离矩阵构造超图关联矩阵
    :param dis_mat: node distance matrix  节点距离矩阵
    :param k_neig: K nearest neighbor  K最近邻
    :param is_probH: prob Vertex-Edge matrix or binary  
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node  从每个节点的中心特征空间构造超边
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):
        dis_mat[center_idx, center_idx] = 0
        dis_vec = dis_mat[center_idx]
        nearest_idx = np.array(np.argsort(dis_vec)).squeeze()
        avg_dis = np.average(dis_vec)
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        for node_idx in nearest_idx[:k_neig]:
            if is_probH:
                H[node_idx, center_idx] = np.exp(-dis_vec[0, node_idx] ** 2 / (m_prob * avg_dis) ** 2)
            else:
                H[node_idx, center_idx] = 1.0
    return H


def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1, mode="global",
                         total_node_num=18, S=8):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix  从原始节点特征矩阵初始化多尺度超图顶点边矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    dis_mat_max = dis_mat.max()
    if mode in ['head','trunk','leg']:
        avail_index = body_id_dict[mode]
        # iter over the dis_mat, set all of the distance that not allowed to max(dist_mat)
        for idx in range(dis_mat.shape[0]):
            for jdx in range(dis_mat.shape[1]):
                if ((idx % total_node_num) not in avail_index) or ((jdx % total_node_num) not in avail_index):
                    dis_mat[idx, jdx] = dis_mat_max

    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H

def __construct_H_with_KNN__(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1, mode="global",
                         total_node_num=18, S=8):
    """
    init multi-scale hypergraph Vertex-Edge matrix from original node feature matrix  从原始节点特征矩阵初始化多尺度超图顶点边矩阵
    :param X: N_object x feature_number
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object x N_hyperedge
    """
    if len(X.shape) != 2:
        X = X.reshape(-1, X.shape[-1])
    if type(K_neigs) == int:
        K_neigs = [K_neigs]

    dis_mat = Eu_dis(X)
    dis_mat_max = dis_mat.max()
    if mode in ['head','trunk','leg']:
        avail_index = body_id_dict[mode]
        # iter over the dis_mat, set all of the distance that not allowed to max(dist_mat)
        for idx in range(dis_mat.shape[0]):
            for jdx in range(dis_mat.shape[1]):
                if ((idx % total_node_num) not in avail_index) or ((jdx % total_node_num) not in avail_index):
                    dis_mat[idx, jdx] = dis_mat_max

    H = []
    for k_neig in K_neigs:
        H_tmp = construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH, m_prob)
        if not split_diff_scale:
            H = hyperedge_concat(H, H_tmp)
        else:
            H.append(H_tmp)
    return H
