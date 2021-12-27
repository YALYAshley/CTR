from __future__ import absolute_import
from __future__ import division
import math
from torch.nn.parameter import Parameter
from HGNN import HGNN_conv
from HGNN import construct_H_with_KNN, generate_G_from_H, hyperedge_concat

__all__ = ['vmgn_hgnn']

import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.model_zoo as model_zoo

from torchreid.utils.reidtools import calc_splits
from torchreid.utils.torchtools import weights_init_kaiming, weights_init_classifier

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):
    """
    Residual network
    Reference:
    He et al. Deep Residual Learning for Image Recognition. CVPR 2016.
    """

    def __init__(self, block, layers, last_stride=2):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class GSTA_HGNN(nn.Module):
    def __init__(self, num_classes, loss, block, layers,
                 num_split, pyramid_part, num_gb, use_pose, learn_graph,
                 consistent_loss, nonlinear='relu',
                 m_prob=1.0, seq_len=8, K_neigs=[10], global_K_neigs=5,is_probH=True,
                 n_hid=128,dropout=0.5,learn_attention=True,learn_edge=False,
                 crop_scale_h=0.5,crop_scale_w=0.5,mode='multi_graph',
                 node_num=[3,4,5],global_branch=False,**kwargs):
        self.inplanes = 64
        super(GSTA_HGNN, self).__init__()
        self.loss = loss
        self.feature_dim = 512 * block.expansion
        self.use_pose = use_pose
        self.learn_graph = learn_graph
        self.learn_attention = learn_attention
        self.learn_edge = learn_edge
        self.seq_len = seq_len
        self.crop_scale_h = crop_scale_h
        self.crop_scale_w = crop_scale_w
        self.mode = mode
        self.node_num = node_num
        self.global_branch = global_branch
        self.num_gb = num_gb
        self.global_edge_weight = None
        self.local_edge_weight = None
        if self.learn_edge:
            total_node_num = sum(node_num)
            if self.mode in ['all_points','part_points','all_graph']:
                global_edge_num = len(global_K_neigs) * seq_len * total_node_num  # the number of edges in the global graphs
                self.global_edge_weight = Parameter(torch.Tensor(np.ones(global_edge_num)))
            if self.mode in ['multi_graph','all_graph']:
                local_edge_num = len(K_neigs) * 3 * seq_len * total_node_num  # the number of edges in the local graphs
                self.local_edge_weight = Parameter(torch.Tensor(np.ones(local_edge_num)))

        # backbone network
        backbone = ResNetBackbone(block, layers, 1)
        init_pretrained_weights(backbone, model_urls['resnet50'])
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4_1 = backbone.layer4
        self.layer4_2 = copy.deepcopy(self.layer4_1)

        # global branch, from layer4_1
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.global_bottleneck.bias.requires_grad_(False)
        self.global_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        weights_init_kaiming(self.global_bottleneck)
        weights_init_classifier(self.global_classifier)

        # split branch, from layer4_2
        self.num_split = num_split
        self.total_split_list = calc_splits(num_split) if pyramid_part else [num_split]
        self.total_split = sum(self.total_split_list)

        # parts branch
        self.parts_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # hgnn graph layers
        self.m_prob = m_prob  # parameter in hypergraph incidence matrix construction
        self.K_neigs = K_neigs  # the number of neighbor expansion
        self.global_K_neigs = global_K_neigs # the number of neighbor for global graph
        self.is_probH = is_probH  # probability Vertex-Edge matrix or binary
        self.n_hid = n_hid
        self.dropout = dropout
        self.hgc_list = nn.ModuleList() # the list of hypergraph layers
        for i in range(self.num_gb):
            self.hgc_list.append(HGNN_conv(self.feature_dim, self.feature_dim))


        # attention branch
        if self.learn_attention:
            self.attention_weight = Parameter(torch.Tensor(self.seq_len, self.total_split,1))
            self._reset_attention_parameters()

        self.consistent_loss = consistent_loss

        self.att_bottleneck = nn.BatchNorm1d(self.feature_dim) # nn.BatchNorm1d(num_classes)
        self.att_bottleneck.bias.requires_grad_(False)
        if self.mode == "all_graph":
            self.att_classifier = nn.Linear(self.feature_dim * 2, num_classes, bias=False)
        else:
            self.att_classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
        weights_init_kaiming(self.att_bottleneck)
        weights_init_classifier(self.att_classifier)

    def _attention_op(self, feat):
        """
        do attention fusion
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        att = F.normalize(feat.norm(p=2, dim=3, keepdim=True), p=1, dim=1)
        f = feat.mul(att).sum(dim=1)
        return f

    def _learn_attention_op(self,feat):
        """
        do attention fusion, with the weight learned
        :param feat: (batch, seq_len, num_split, c)
        :return: feat: (batch, num_split, c)
        """
        f = feat.mul(self.attention_weight)
        f = f.sum(dim=1)
        return f

    def _reset_attention_parameters(self):
        stdv = 1. / math.sqrt(self.attention_weight.size(1))
        self.attention_weight.data.uniform_(-stdv, stdv)
        # nn.init.normal_(self.attention_weight.data, 0.1,0.001)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x4_1 = self.layer4_1(x)
        x4_2 = self.layer4_2(x)
        return x4_1, x4_2

    def propogation_attention(self, f, B, S, G, total_node_num):
        """
        implement the graph propagation abd attention mechanism
        :return: att_f
        """
        # hgnn graph propogation
        for i in range(self.num_gb - 1):
            f = F.relu(self.hgc_list[i](f, G))
            f = F.dropout(f, self.dropout)
        f = self.hgc_list[-1](f, G)

        f = f.view(B, S, total_node_num, f.shape[-1])

        # attention branch
        if self.learn_attention:  # learn the weight of attention
            f_fuse = self._learn_attention_op(f)
        else:  # calculate the norm as the attention weight
            f_fuse = self._attention_op(f)

        att_f = f_fuse.mean(dim=1).view(B, -1)
        return att_f

    def forward(self, x, pose_loc, *args):
        B, S, C, H, W = x.size()
        x = x.view(B * S, C, H, W) # change the size of tensor
        x4_1, x4_2 = self.featuremaps(x) # features of the input sequence
        _, c, h, w = x4_1.shape

        # global branch
        if self.global_branch:
            x4_1 = x4_1.view(B, S, c, h, w).transpose(1, 2).contiguous()
            g_f = self.global_avg_pool(x4_1).view(B, -1)
            g_bn = self.global_bottleneck(g_f)

        # crop branch
        B, S, total_node_num, p = pose_loc.shape
        pose_loc = pose_loc.view(B * S, total_node_num, p)
        crop_size_h = int(self.crop_scale_h * x4_2.shape[-1])
        crop_size_w = int(self.crop_scale_w * x4_2.shape[-2])

        f = []
        for idx in range(len(x4_2)): # iter over each batch and sequence image
            feature = x4_2[idx]
            nodes = []
            for jdx in range(total_node_num):
                (left, top) = pose_loc[idx][jdx]
                left, top = int(left * x4_2.shape[-1]), int(top * x4_2.shape[-2])
                left = min(max(left, 0), x4_2.shape[-1] - crop_size_h - 1)
                top = min(max(top, 0), x4_2.shape[-2] - crop_size_w - 1)
                split = feature[:,top:top+crop_size_w,left:left+crop_size_h]
                nodes.append(split)
            nodes = torch.stack(nodes)
            f.append(nodes)
        f = torch.stack(f)

        f = f.view(B * S * total_node_num, c, crop_size_w, crop_size_h)
        f = self.parts_avgpool(f)
        f = f.view(B, S * total_node_num, c)

        # construct the hgnn graph
        if self.mode in ['all_points','part_points','multi_graph']:
            G = []
            for feature in f:  # iter over each batch
                if self.mode in ['all_points','part_points']:
                    H_global = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.global_K_neigs, is_probH=self.is_probH,
                                         m_prob=self.m_prob, mode="global", total_node_num=total_node_num, S=S)
                    g = generate_G_from_H(H_global, variable_weight=self.learn_edge,edge_weight=self.global_edge_weight)
                elif self.mode == "multi_graph":
                    H = None
                    H_head = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs,
                                                    is_probH=self.is_probH,
                                                    m_prob=self.m_prob, mode="head", total_node_num=total_node_num, S=S)
                    H = hyperedge_concat(H, H_head)
                    H_trunk = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs,
                                                  is_probH=self.is_probH,
                                                  m_prob=self.m_prob, mode="trunk", total_node_num=total_node_num, S=S)
                    H = hyperedge_concat(H, H_trunk)
                    H_leg = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs,
                                                  is_probH=self.is_probH,
                                                  m_prob=self.m_prob, mode="leg", total_node_num=total_node_num, S=S)
                    H = hyperedge_concat(H, H_leg)
                    g = generate_G_from_H(H, variable_weight=self.learn_edge,edge_weight=self.local_edge_weight)
                G.append(g)
            G = torch.stack(G)

            att_f = self.propogation_attention(f,B,S,G,total_node_num)
            att_bn = self.att_bottleneck(att_f)
        elif self.mode == 'all_graph':
            G_local = []
            G_global = []
            for feature in f:  # iter over each batch
                H = None
                H_head = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs,
                                              is_probH=self.is_probH,
                                              m_prob=self.m_prob, mode="head", total_node_num=total_node_num, S=S)
                H = hyperedge_concat(H, H_head)
                H_trunk = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs,
                                               is_probH=self.is_probH,
                                               m_prob=self.m_prob, mode="trunk", total_node_num=total_node_num, S=S)
                H = hyperedge_concat(H, H_trunk)
                H_leg = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.K_neigs,
                                             is_probH=self.is_probH,
                                             m_prob=self.m_prob, mode="leg", total_node_num=total_node_num, S=S)
                H_local = hyperedge_concat(H, H_leg)
                H_global = construct_H_with_KNN(feature.cpu().detach().numpy(), K_neigs=self.global_K_neigs,
                                                is_probH=self.is_probH,
                                                m_prob=self.m_prob, mode="global", total_node_num=total_node_num, S=S)
                g_local = generate_G_from_H(H_local, variable_weight=self.learn_edge, edge_weight=self.local_edge_weight)
                g_global = generate_G_from_H(H_global, variable_weight=self.learn_edge, edge_weight=self.global_edge_weight)
                if self.learn_edge:
                    G_local.append(g_local)
                    G_global.append(g_global)
                else:
                    G_local.append(g_local)
                    G_global.append(g_global)
            if self.learn_edge:
                G_local = torch.stack(G_local)
                G_global = torch.stack(G_global)
            else:
                G_local = torch.stack(G_local)
                G_global = torch.stack(G_global)
            att_f_local = self.propogation_attention(f, B, S, G_local, total_node_num)
            att_f_global = self.propogation_attention(f, B, S, G_global, total_node_num)

            # concat two features
            att_bn_local = self.att_bottleneck(att_f_local)
            att_bn_global = self.att_bottleneck(att_f_global)
            att_f = torch.cat([att_f_global, att_f_local], dim=1)
            att_bn = torch.cat([att_bn_global, att_bn_local], dim=1)

        if not self.training:
            if self.global_branch:
                return torch.cat([g_bn, att_bn], dim=1)
            else:
                return att_bn

        if self.global_branch:
            g_out = self.global_classifier(g_bn)
        att_out = self.att_classifier(att_bn)

        if self.loss == {'xent'}:
            out_list = [g_out, att_out] if self.global_branch else [att_out]
            return out_list
        elif self.loss == {'xent', 'htri'}:
            out_list = [g_out, att_out] if self.global_branch else [att_out]
            f_list = [g_f, att_f] if self.global_branch else [att_f]
            return out_list, f_list
        else:
            raise KeyError('Unsupported loss: {}'.format(self.loss))


def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


def vmgn_hgnn(num_classes, loss, last_stride, num_split, num_gb, num_scale,
         pyramid_part, use_pose, learn_graph, pretrained=True, consistent_loss=False,
         m_prob=1.0, seq_len=8, K_neigs=[10],global_K_neigs=5, is_probH=True,
         n_hid=128,dropout=0.5,learn_attention=True,learn_edge=False,
         crop_scale_h=0.5,crop_scale_w=0.5,mode='multi_graph',node_num=[3,4,5],
         global_branch=False, **kwargs):
    model = GSTA_HGNN(
        num_classes=num_classes,
        loss=loss,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        last_stride=last_stride,
        num_split=num_split,
        pyramid_part=pyramid_part,
        num_gb=num_gb,
        use_pose=use_pose,
        learn_graph=learn_graph,
        consistent_loss=consistent_loss,
        nonlinear='relu',
        m_prob=m_prob,
        seq_len=seq_len,
        K_neigs=K_neigs,
        global_K_neigs=global_K_neigs,
        is_probH=is_probH,
        n_hid=n_hid,
        dropout=dropout,
        learn_attention=learn_attention,
        learn_edge=learn_edge,
        crop_scale_h=crop_scale_h,
        crop_scale_w=crop_scale_w,
        mode=mode,
        node_num=node_num,
        global_branch=global_branch,
        **kwargs
    )

    return model