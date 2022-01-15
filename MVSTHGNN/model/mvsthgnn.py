from HGNN import HGNN_conv
import torch.nn.functional as F
from torch import nn

class FrameWiseHGNN(nn.Module):
    def __init__(self, hids=[3, 32, 64, 128], class_num=8):
        super(FrameWiseHGNN, self).__init__()
        self.activation = nn.Softmax(dim=1)
        # hgnn layers
        self.layers = nn.ModuleList()  # the list of hypergraph layers
        for j in range(1, len(hids)):
            self.layers.append(HGNN_conv(hids[j - 1], hids[j]))  # set hiddens
        self.cls_layer = nn.Linear(hids[-1], class_num)  # 设置全链接层，输入格式是1*8
        print("cls_layer:", self.cls_layer)

    def forward(self, x, G):

        for L in self.layers:
            x = L(x, G)  # 进入网络层
            x = F.leaky_relu(x)
            x = F.dropout(x, training=self.training)
        # bz * n * c -> bz * c
        x = x.max(1)[0]
        # 1 x c -> 1 x 8
        out_feature = self.cls_layer(x)

        # result = self.activation(out_feature)
        result = out_feature
        return result