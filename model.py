import dgl
from dgl.nn import GraphConv

import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_features, h_features, num_classes):
        '''
        in_features: dimension of node feature
        h_features:  dimension of hidden feature
        num_classes: dimension of output feature, number of total classes
        '''
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_features, h_features)
        self.conv2 = GraphConv(h_features, num_classes)

    def forward(self, g, x):
        h = self.conv1(g, x)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h



