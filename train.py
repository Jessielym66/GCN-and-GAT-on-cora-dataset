import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import dgl
from dgl.nn import GraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import tensorboard

from model import GCN
import argparse

parser = argparse.ArgumentParser(description = 'PyTorch GCN')
parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--logs', type=str, default='./logs', help='the path to save model and results')
args = parser.parse_args()

def train(model, g, optimizer, args):
    # graph
    node_features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']

    writer = tensorboard.SummaryWriter(log_dir=args.logs)
    
    for epoch in range(1, args.epochs+1):
        # train
        model.train()
        logits = model(g, node_features)
        pred = logits.argmax(1)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # val
        model.eval()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        print("Epoch {}\t train_loss: {:.3f}\t val_acc: {:.3f}%".format(epoch, loss, val_acc*100))
        
        writer.add_scalar('Loss/train_loss', loss.cpu().detach().numpy(), global_step=epoch)
        writer.add_scalar('Acc/val_accuracy', val_acc.cpu().detach().numpy(), global_step=epoch)
    
    torch.save(model.state_dict(), os.path.join(args.logs, 'model.pkl'))



def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("use {}".format(device))
    if not os.path.exists(args.logs):
        os.mkdir(args.logs)

    # load Cora dataset
    dataset = dgl.data.CoraGraphDataset()
    g = dataset[0]
    g = g.to(device)

    # create GCN model
    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # train model
    train(model, g, optimizer, args)


if __name__ == '__main__':
    main()