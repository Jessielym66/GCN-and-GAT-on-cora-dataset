import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GCN
import argparse

parser = argparse.ArgumentParser(description = 'PyTorch GCN')
parser.add_argument('--checkpoints', type=str, default='./logs', help='the path to load model')
args = parser.parse_args()

def test(model, g):
    # graph
    node_features = g.ndata['feat']
    labels = g.ndata['label']
    test_mask = g.ndata['test_mask']

    model.eval()
    logits = model(g, node_features)
    pred = logits.argmax(1)
    test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
    
    print("test accuracy: {:.3f}".format(test_acc))
   
        

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load Cora dataset
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0].to(device)

    # create GCN model
    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to(device)
    state_dict = torch.load(os.path.join(args.checkpoints, 'model.pkl'))
    model.load_state_dict(state_dict)

    test(model, g)


if __name__ == '__main__':
    main()