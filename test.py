import os, sys, torch, dgl
import numpy as np
import pandas as pd
from util import check_files
from data_ import *
from train import *
from torch_geometric.data import Data

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
# from dgl.nn import Set2Set

from torch_geometric.nn import SAGEConv, SAGPooling, Set2Set, GraphNorm, global_sort_pool, GlobalAttention
from torch_geometric.utils import add_self_loops, subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

omics_files=["./data/LGG/rna.csv.gz", "./data/LGG/met.csv.gz", "./data/LGG/miRNA_gene_mean.csv.gz"]
label_file = "./data/LGG/label.csv"
add_file = None
G, pathway, id_mapping = read_omics(omics_files=omics_files, 
                                    label_file=label_file, 
                                    add_file=add_file, 
                                    pathway_file='./Pathway/Rectome.pathway.tmp.csv')

labels = G.label


class PathFeature(nn.Module):
    def __init__(self, in_dim):
        super(PathFeature, self).__init__()
        # GNN-1
        self.conv1 = SAGEConv(in_dim, in_dim)
        self.pool1 = SAGPooling(in_dim, ratio=0.8)
        self.readout1 = GlobalAttention(gate_nn=nn.Linear(in_dim, 1))
        
    def forward(self, g, h):
        edge_index = g.edge_index
        x = h
        # GNN-1
        x = torch.relu(self.conv1(x, edge_index))
        x, edge_index, _, _, _, _ = self.pool1(x, edge_index, None, None)
        x = self.readout1(x)

        return x


class DeepMOI(nn.Module):
    def __init__(self, in_dim, pathway, add_features=None):
        """
        in_dim: == omics' number
        hidden_dim: == 
        """
        super(DeepMOI, self).__init__()
        self.pathway = pathway
        
        # Gene Layer
        self.conv = SAGEConv(in_dim, in_dim, 'pool')
                
        # Pathway Layers     
        self.submodels = nn.ModuleList()
        for _ in range(len(self.pathway.pathway.unique())):
            self.submodels.append(PathFeature(in_dim=in_dim))
    
        # MLP
        self.lin = nn.Linear(3, 1)
        self.mlp = nn.Sequential(
                                 nn.Dropout(p=0.2),
                                 nn.ReLU(),
                                 nn.Linear(len(self.pathway.pathway.unique()), 1),
                                 nn.Sigmoid()
                                )
        
    def forward(self, g, h, c=None, output=False):
        edge_index = g.edge_index
        edge_index,_ = add_self_loops(edge_index=edge_index)
        x = h
        
        # Gene Layer
        x = torch.tanh(self.conv(x, edge_index))    
        
        # Pathway Layer
        i = 0
        readout = []
        for path, group in self.pathway.groupby('pathway'):
            nodes = list(set(group.src.to_list() + group.dest.to_list()))
            sub_edge_idx,_ = subgraph(subset=nodes, edge_index=edge_index)
            dat = Data(x=x, edge_index=sub_edge_idx)
            out = self.submodels[i](dat, x)
            out = torch.relu(out)
            readout.append(out)
        readout = torch.cat(readout, dim=0)
        readout = torch.relu(self.lin(readout).T)
        if output:
            return readout
        # MLP
        logit = self.mlp(readout)

        return logit


np.random.seed(42)
epoch = 100
lr = 0.0005
minibatch=16
device='cpu'
outdir='./'
train_idx, test_idx = data_split(labels=G.label, test_size=0.3)

model = DeepMOI(in_dim=3, pathway=pathway)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.00001)

# train model
for epoch in range(epoch):
    model.train()
    logits_epoch, labels_epoch, loss_epoch = [], [], [] # for training dataset evaluation
    for idx in batch_idx(train_idx=train_idx, minibatch=minibatch):
        logits_batch = []
        for i in idx:
            logit = model(G, G.x[:, i, :])
            logits_batch.append(logit)
            logits_epoch.append(logit.to(device='cpu').detach().numpy())
            print("epoch {}  | label {}  |".format(epoch, labels[i]), logit)
        # backward
        loss = nn.BCELoss()(torch.cat(logits_batch), torch.tensor(labels[idx], dtype=torch.float32, device=device).reshape(-1,1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_epoch.append(loss.item())
        labels_epoch += idx

    torch.save(model, os.path.join(outdir,'model/DeepMOI_{}.pt'.format(epoch)))
print('1st finished')

##############
##############
##############
model.eval()
outputs = []
labels=G.label
for i in range(len(labels)):
    print(i)
    outputs.append(model(G, G.x[:, i, :], output=True))

import dgl
from dgl.nn.pytorch.factory import KNNGraph

x = G.x.permute(1,0,2)
kg = KNNGraph(10)
sample_graph = kg(x[:, :, 0])
sample_graph.ndata['label'] = torch.tensor(labels, dtype=torch.float32).reshape(-1,1)
sample_graph.ndata['x'] = torch.cat(outputs).detach()

# 构建一个2层的GNN模型
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        # 实例化SAGEConve，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregator_type是聚合函数的类型
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # 输入是节点的特征
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
#         h = torch.sigmoid(h)
        return h

ll = torch.tensor(G.label, dtype=torch.long)
model = SAGE(in_feats=115, hid_feats=64, out_feats=2)
opt = torch.optim.Adam(model.parameters())

for epoch in range(50):
    model.train()
    # 使用所有节点(全图)进行前向传播计算
    logits = model(sample_graph, sample_graph.ndata['x'])
    
    # 计算损失值
    loss = F.cross_entropy(logits[train_idx], ll[train_idx] )
    
    # 进行反向传播计算
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("Epoch {} || ".format(epoch),loss.item())
    pred = logits.argmax(1)
    train_acc = np.sum(pred[train_idx].numpy()==labels[train_idx]) / 509
    test_acc = np.sum(pred[test_idx].numpy()==labels[test_idx]) / 509
    print("ACC: ",train_acc, test_acc)
