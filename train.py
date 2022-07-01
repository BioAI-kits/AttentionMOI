import sys, os, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import Set2Set, SumPooling
from data import read_omics, read_clin, build_graph, read_pathways
from util import evaluate, check_files
from module import DeepMOI


warnings.filterwarnings('ignore')
np.random.seed(1234)


def batch_idx(train_idx, minibatch):
    """To obtain batch index.
    train_idx (list): index of training dataset.
    minibatch (int): sample's number in each batch.

    Return:
        batch_idx (list): the element is list, i.e., index for each batch.
    """
    batch_idx, m = [], 0
    while True:
        if (m+1)*minibatch < len(train_idx):
            batch_idx.append(train_idx[m*minibatch:(m+1)*minibatch])
        else:
            batch_idx.append(train_idx[m*minibatch:])
            break
        m += 1
    return batch_idx


def data_split(labels, test_size):
    """To split dataset into training dataset and testing dataset.
    Args:
        labels (tensor): The labels of samples.
        test_size (float, 0-1): The proportion of test data.
    
    Return:
        (list) index of training data, index of testing data
    """
    test_number = int(len(labels) * test_size)
    idx = list(range(len(labels)))
    np.random.shuffle(idx)
    return idx[test_number:], idx[:test_number]



def train(omics_files, clin_file, test_size, pathway_file='default', minibatch=16, epoch=10, lr=0.001):
    # read dataset
    print('[INFO] Reading dataset. There are {} omics data in total.'.format(len(omics_files)))
    omics = read_omics(omics_files=omics_files, clin_file= clin_file)
    graph, labels, clin_features, id_mapping = build_graph(omics=omics, clinical_file=clin_file)

    # split dataset
    train_idx, test_idx = data_split(labels=labels, test_size=test_size)

    # read pathway
    if pathway_file == 'default':
        base_path = os.path.split(os.path.realpath(__file__))[0]
        pathway_file = os.path.join(base_path, 'Pathway', 'pathway_genes.gmt')
    pathways = read_pathways(id_mapping=id_mapping, file=pathway_file)

    # init model
    print('[INFO] Training model.')
    model = DeepMOI(in_dim=len(omics_files), pathway=pathways, clinical_feature=clin_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # train model
    for epoch in range(epoch):
        model.train()
        optimizer.zero_grad()
        logits_epoch, labels_epoch, loss_epoch = [], [], [] # for training dataset evaluation
        for idx in batch_idx(train_idx=train_idx, minibatch=minibatch):
            logits_batch = []
            # patients-wise
            for i in idx:
                
                logit = model(g=graph, h=graph.ndata['h'][:, i, :], a=)







    # graphs_train, graphs_test, lables_train, labels_test, clin_features_train, clin_features_test = train_test_split(graphs, labels, clin_features, test_size=0.2, random_state=42)
    graphs_train = np.array(graphs_train)
    graphs_test = np.array(graphs_test)

    
    # train model
    for epoch in range(200):
        model.train()
        opt.zero_grad()
        logits_epoch, labels_epoch, loss_epoch = [], [], [] # for training dataset evaluation
        for idx in batch_idx(graphs=graphs_train, minibatch=minibatch):
            logits_batch = []
            # patients-wise
            for i in idx:
                graph, label = graphs_train[i], lables_train[i].unsqueeze(0).float()
                clinical_feature = clin_features_train[i].reshape(1,-1).squeeze(0).to(torch.float32)
                logit = model(graph, graph.ndata['h'], clinical_feature)
                logits_batch.append(logit)
                logits_epoch.append(logit.detach().numpy())
            # backward
            loss = nn.BCELoss()(torch.cat(logits_batch), lables_train[idx].to(torch.float32))
            loss.backward()
            opt.step()
            loss_epoch.append(loss.item())
            labels_epoch += idx

        # evaluation for training dataset
        logits_epoch = np.concatenate(logits_epoch)
        labels_epoch = labels[labels_epoch].detach().numpy()
        loss_epoch = np.mean(loss_epoch)
        acc, auc, f1_score_, sens, spec = evaluate(logits=logits_epoch, real_labels=labels_epoch)
        print('Epoch {:2d} | Loss {:.10f} | Acc {:.3f} | AUC {:.3f} | F1_score {:.3f} | Sens {:.3f} | Spec {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
            )
        with open('log.txt', 'a') as F:
            F.writelines('Epoch {:2d} | Loss {:.10f} | Acc {:.3f} | AUC {:.3f} | F1_score {:.3f} | Sens {:.3f} | Spec {:.3f}\n'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec))
        
        # evaluation for testing dataset
        if epoch > 100:
            model.eval()
            logits = []
            for i in range(len(graphs_test)):
                g = graphs_test[i]
                c = clin_features_test[i].reshape(1,-1).squeeze(0).to(torch.float32)
                logit = model(g, g.ndata['h'], c)
                logits.append(logit.detach().numpy())
            logits = np.concatenate(logits)
            acc, auc, f1_score_, sens, spec = evaluate(logits, labels_test.detach().numpy())
            print('Test_Epoch {:2d} | Test_Loss {:.5f} | Test_Acc {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Sens {:.3f} | Test_Spec {:.3f}'.format(
                    epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
                    )
            with open('log.txt', 'a') as F:
                F.writelines('Test_Epoch {:2d} | Test_Loss {:.5f} | Test_Acc {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Sens {:.3f} | Test_Spec {:.3f}\n'.format(
                epoch, loss_epoch, acc, auc, f1_score_, sens, spec))
        
        # save model
        if epoch % 5 == 0:
            torch.save(model, './model/DeepMOI_{}.pt'.format(epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--omic_file', action='append', help='omics file.', required=True)
    parser.add_argument('-c','--clin', help='clinical file.', required=True)
    parser.add_argument('-l','--label', help='label file', required=True)
    parser.add_argument('-b','--batch', help='Mini-batch number.', type=int, default=16)
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=10)
    parser.add_argument('-p','--pathway', help='The pathway file that should be gmt format.', type=str, default='default')
    args = parser.parse_args()
    
    # check files exists
    check_files(args.omic_file)
    check_files(args.clin_file)

    # Running main function
    main(omics_files=args.omic_file, clin_file=args.clin_file, minibatch=args.batch, epoch=args.epoch)

    
    print("[INFO] DeepMOI has finished running!")

