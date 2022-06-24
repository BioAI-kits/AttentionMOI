import sys, os, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import SumPooling
from data import read_omics, read_clin, build_graph, read_pathways
from util import evaluate, check_files

np.random.seed(1234)


class DeepMOI(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, pathway):
        super(DeepMOI, self).__init__()
        self.gin_lin1 = torch.nn.Linear(in_dim, hidden_dim)
        self.conv1 = dglnn.GINConv(self.gin_lin1, 'sum')

        self.gin_lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.conv2 = dglnn.GINConv(self.gin_lin2)
        
        self.lin1 = nn.Linear(hidden_dim*2, 1)
        self.lin2 = nn.Linear(len(pathway), 2)
        self.pathway = pathway

    def forward(self, g, h):
        # subnetwork1: GRL layers
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        # subnetwork2: patyway layers
        with g.local_scope():
            g.ndata['h'] = h
            # unbatch
            g_unbatch = dgl.unbatch(g)
            logits = []
            for g in g_unbatch:
                # global pooling 1
                subgraphs = [dgl.node_subgraph(g, n) for n in self.pathway.values()]        
                h_mean = [dgl.mean_nodes(g, 'h') for sg in subgraphs]
                h_mean = torch.cat(h_mean)
                # global pooling 2
                sumpool = SumPooling()
                h_sumpool = [sumpool(sg, sg.ndata['h']) for sg in subgraphs]
                h_sumpool = torch.cat(h_sumpool)
                # concat global pooling
                h = torch.cat([h_mean, h_sumpool], 1)
                # linear-1
                h = F.tanh(self.lin1(h).squeeze(1))
                # classification
                logit = F.softmax(self.lin2(h))
                logits.append(logit)
            return torch.stack(logits, 0)


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


def main(omics_files, clin_file, minibatch=16, epoch=10, pathway_file='default'):
    # warnings.filterwarnings('ignore')
    print('[INFO] Reading dataset.')
    omics = read_omics(omics_files=omics_files, clin_file= clin_file)
    graphs, labels, clin_features, id_mapping = build_graph(omics=omics, clinical_file=clin_file)
    graphs = np.array(graphs)

    # read pathways
    if pathway_file == 'default':
        base_path = os.path.split(os.path.realpath(__file__))[0]
        pathway_file = os.path.join(base_path, 'Pathway', 'pathway_genes.gmt')
    pathways = read_pathways(id_mapping=id_mapping, file=pathway_file)

    # init model
    print('[INFO] Training model.')
    model = DeepMOI(in_dim=3, hidden_dim=8, n_classes=2, pathway=pathways)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(200):
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
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=16)
    parser.add_argument('-p','--pathway', help='The pathway file that should be gmt format.', type=str, default='default')
    args = parser.parse_args()
    
    # check files exists
    check_files(args.omic_file)
    check_files(args.clin_file)

    # Running main function
    main(omics_files=args.omic_file, clin_file=args.clin_file, minibatch=args.batch, epoch=args.epoch)

    
    print("Finished!")

