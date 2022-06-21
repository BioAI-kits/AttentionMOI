import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from data import GraphOmics


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # 应用图卷积和激活函数
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # 使用平均读出计算图表示
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)


def main():
    omics_files = ['./data/GBM/GBM.cnv.csv.gz', 
                   './data/GBM/GBM.expression.csv.gz', 
                   './data/GBM/GBM.met.csv.gz']
    clin_file= './data/GBM/clinincal.csv'
    dataloader = GraphOmics(omics_fliles=omics_files, clin_file=clin_file)
    model = Classifier(in_dim=3, hidden_dim=8, n_classes=2)
    opt = torch.optim.Adam(model.parameters())

    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['h']
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
    print("Epoch: \n", epoch)

main()





