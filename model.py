import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from data import read_omics, read_clin, build_graph
from util import evaluate, check_files

np.random.seed(1234)


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


def batch_idx(graphs, minibatch=16):
    """To obtain batch index.
    graphs (list): the element is graph.
    minibatch (int, default=8): graph number in each batch.

    Return:
        batch_idx (list): the element is list, i.e., index for each batch.
    """

    idx = list(range(len(graphs)))
    np.random.shuffle(idx)
    batch_idx, m = [], 0
    while True:
        if (m+1)*minibatch < len(graphs):
            batch_idx.append(idx[m*minibatch:(m+1)*minibatch])
        else:
            batch_idx.append(idx[m*minibatch:])
            break
        m += 1
    return batch_idx


def main(omics_files, clin_file, minibatch=16, epoch=10):
    print('[INFO] Reading dataset.')
    omics = read_omics(omics_files=omics_files, clin_file= clin_file)
    graphs, labels, clin_features, id_mapping = build_graph(omics=omics, clinical_file=clin_file)
    graphs = np.array(graphs)

    # init model
    print('[INFO] Training model.')
    model = Classifier(in_dim=3, hidden_dim=8, n_classes=2)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        logits_epoch, labels_epoch, loss_epoch = [], [], [] # for training dataset evaluation
        for idx in batch_idx(graphs=graphs, minibatch=minibatch):            
            batched_graph = dgl.batch(graphs[idx])
            feats = batched_graph.ndata['h']
            # fill nan with 0
            feats = torch.where(feats.isnan(), torch.full_like(feats, 0), feats)
            batched_labels = labels[idx]

            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, batched_labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            logits_epoch.append(logits)
            labels_epoch.append(batched_labels)
            loss_epoch.append(loss.item())

        # evaluation for training dataset
        logits_epoch = torch.cat(logits_epoch, 0)
        labels_epoch = torch.cat(labels_epoch, 0)
        logits_epoch = logits_epoch.detach().numpy()
        labels_epoch = labels_epoch.detach().numpy()
        loss_epoch = np.mean(loss_epoch)
        acc, auc, f1_score_, sens, spec = evaluate(logits=logits_epoch, real_labels=labels_epoch)
        print('Epoch {:2d} | Loss {:.5f} | Acc {:.3f} | AUC {:.3f} | F1_score {:.3f} | Sens {:.3f} | Spec {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--omic_file', action='append', help='omics file.', required=True)
    parser.add_argument('-c','--clin_file', help='clinical file.', required=True)
    parser.add_argument('-b','--batch', help='Mini-batch number.', type=int, default=8)
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=10)
    args = parser.parse_args()
    
    # check files exists
    check_files(args.omic_file)
    check_files(args.clin_file)

    # Running main function
    main(omics_files=args.omic_file, clin_file=args.clin_file, minibatch=args.batch, epoch=args.epoch)

    
    print("Finished!")


#! TODO
# add graph layers for model.

