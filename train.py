import sys, os, argparse, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from util import *
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
        labels (numpy): The labels of samples.
        test_size (float, 0-1): The proportion of test data.
    
    Return:
        (list) index of training data, index of testing data
    """
    test_number = int(len(labels) * test_size)
    idx = list(range(len(labels)))
    np.random.shuffle(idx)
    return idx[test_number:], idx[:test_number]



def train(omics_files, label_file, add_file, test_size, pathway_file, network_file, minibatch, epoch, lr):
    """ To train DeepMOI model.

    Args:
        omics_files (list, required):   omic filenames.
        label_file (str, required) :    label filename.
        add_file (str, optional) :      additional features' filename.
        test_size (float, 0-1):         proportion of testing dataset.
        pathway_file (str):             pathway filename.
        network_file (str):             network filename.
        minibatch (int):                minibatch size.
        epoch (int):                    epoch number.
        lr (float):                     learning rate.

    Returns:

    """
    # read dataset
    print('[INFO] Reading dataset. There are {} omics data in total.\n'.format(len(omics_files)))
    omics = read_omics(omics_files=omics_files, label_file=label_file, add_file=add_file)
    graph, labels, add_features, id_mapping = build_graph(omics=omics, label_file=label_file, add_file=add_file, network_file=network_file)
    omic_features = graph.ndata['h']

    # read pathway
    if pathway_file == 'default':
        base_path = os.path.split(os.path.realpath(__file__))[0]
        pathway_file = os.path.join(base_path, 'Pathway', 'pathway_genes.gmt')
    pathways = read_pathways(id_mapping=id_mapping, file=pathway_file)

    # split dataset
    train_idx, test_idx = data_split(labels=labels, test_size=test_size)

    # init model
    print('[INFO] Training model.\n')
    model = DeepMOI(in_dim=len(omics_files), pathway=pathways, add_features=add_features)
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
                print('idx:', i)
                if add_features != None:
                    logit = model(g=graph, h=omic_features[:, i, :], c=add_features[i])
                else:
                    logit = model(g=graph, h=omic_features[:, i, :], c=None)
                logits_batch.append(logit)
                logits_epoch.append(logit.detach().numpy())
            # backward
            print(torch.cat(logits_batch), torch.tensor(labels[idx], dtype=torch.float32))
            loss = nn.BCELoss()(torch.cat(logits_batch), torch.tensor(labels[idx], dtype=torch.float32).reshape(-1,1))
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            labels_epoch += idx
        
        # evaluation for training dataset
        logits_epoch = np.concatenate(logits_epoch)
        labels_epoch = labels[train_idx]
        loss_epoch = np.mean(loss_epoch)
        acc, auc, f1_score_, sens, spec = evaluate(logits=logits_epoch, real_labels=labels_epoch)
        print('Epoch {:2d} | Loss {:.10f} | Acc {:.3f} | AUC {:.3f} | F1_score {:.3f} | Sens {:.3f} | Spec {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
            )
        with open('log.txt', 'a') as F:
            F.writelines('Epoch {:2d} | Loss {:.10f} | Acc {:.3f} | AUC {:.3f} | F1_score {:.3f} | Sens {:.3f} | Spec {:.3f}\n'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec))
        
        # evaluation for testing dataset
        if epoch > 3:
            model.eval()
            logits = []
            for i in test_idx:
                print('idx:', i)
                if add_features != None:
                    logit = model(g=graph, h=omic_features[:, i, :], c=add_features[i])
                else:
                    logit = model(g=graph, h=omic_features[:, i, :], c=None)
                logits.append(logit.detach().numpy())
            logits = np.concatenate(logits)
            acc, auc, f1_score_, sens, spec = evaluate(logits, labels[test_idx])
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
    parser.add_argument('-l','--label_file', help='label file', required=True)
    parser.add_argument('-a','--additional_features', default=None, help='Non-omic features')
    parser.add_argument('-p','--pathway', help='The pathway file that should be gmt format.', type=str, default='default')
    parser.add_argument('-n','--network', help='The network file that should be gmt format.', type=str, default='default')
    parser.add_argument('-b','--batch', help='Mini-batch number.', type=int, default=16)
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=10)
    parser.add_argument('-r','--lr', help='learning reate.', type=float, default=0.001)
    args = parser.parse_args()
    
    print(args.omic_file)
    # # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # # Running main function
    train(omics_files=args.omic_file, 
          label_file=args.label_file, 
          add_file=args.additional_features,
          test_size=0.2,
          pathway_file=args.pathway,
          network_file=args.network,
          minibatch=args.batch, 
          epoch=args.epoch,
          lr=args.lr 
          )

    
    # print("[INFO] DeepMOI has finished running!")

