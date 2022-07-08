import sys, os, argparse, warnings, time
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


def train(omics_files, label_file, add_file, test_size, pathway_file, network_file, minibatch, epoch, lr, outdir, device='cpu'):
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
        outdir (path):                  output dir.
    Returns:

    """
    # read dataset
    print('[INFO] Reading dataset. There are {} omics data in total.\n'.format(len(omics_files)))
    omics = read_omics(omics_files=omics_files, label_file=label_file, add_file=add_file)
    graph, labels, add_features, id_mapping = build_graph(omics=omics, label_file=label_file, add_file=add_file, network_file=network_file)
    omic_features = graph.x
    
    # to device
    graph = graph.to(device)
    omic_features = omic_features.to(device)
    if add_features != None:
        add_features = add_features.to(device)
    
    # read pathway
    if pathway_file == 'default':
        base_path = os.path.split(os.path.realpath(__file__))[0]
        pathway_file = os.path.join(base_path, 'Pathway', 'pathway_genes.gmt')
    pathways = read_pathways(id_mapping=id_mapping, file=pathway_file)

    # split dataset
    train_idx, test_idx = data_split(labels=labels, test_size=test_size)

    # init model
    print('[INFO] Start training model:')
    model = DeepMOI(in_dim=len(omics_files), pathway=pathways, add_features=add_features).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.003)

    # train model
    for epoch in range(epoch):
        model.train()

        # # 
        # if epoch < 50:
        #     for k,v in model.named_parameters():
        #         if k.startswith('lin'):
        #             v.requires_grad = False
        # else:
        #     for k,v in model.named_parameters():
        #         if  k.startswith('lin'):
        #             v.requires_grad = True
        #         else:
        #             v.requires_grad = False

        logits_epoch, labels_epoch, loss_epoch = [], [], [] # for training dataset evaluation
        for idx in batch_idx(train_idx=train_idx, minibatch=minibatch):
            logits_batch = []
            # patients-wise
            for i in idx:
                if add_features != None:
                    logit = model(g=graph, h=omic_features[:, i, :], c=add_features[i])
                else:
                    logit = model(g=graph, h=omic_features[:, i, :], c=None)
                logits_batch.append(logit)
                logits_epoch.append(logit.to(device='cpu').detach().numpy())
            # backward
            loss = nn.BCELoss()(torch.cat(logits_batch), torch.tensor(labels[idx], dtype=torch.float32, device=device).reshape(-1,1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_epoch.append(loss.item())
            labels_epoch += idx
        
        # evaluation for training dataset
        logits_epoch = np.concatenate(logits_epoch)
        labels_epoch = labels[train_idx]
        loss_epoch = np.mean(loss_epoch)
        acc, auc, f1_score_, sens, spec = evaluate(logits=logits_epoch, real_labels=labels_epoch)
        print('Epoch {:2d} | Train_Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Sens {:.3f} | Train_Spec {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
            )
        with open(os.path.join(outdir,'log.txt'), 'a') as F:
            F.writelines('Epoch {:2d} | Train_Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Sens {:.3f} | Train_Spec {:.3f}\n'.format(
            epoch, loss_epoch, acc, auc, f1_score_, sens, spec))
        
        # evaluation for testing dataset
        if epoch > 5:
            model.eval()
            logits = []
            for i in test_idx:
                if add_features != None:
                    logit = model(g=graph, h=omic_features[:, i, :], c=add_features[i])
                else:
                    logit = model(g=graph, h=omic_features[:, i, :], c=None)
                logits.append(logit.detach().numpy())
            logits = np.concatenate(logits)
            acc, auc, f1_score_, sens, spec = evaluate(logits, labels[test_idx])
            print('Epoch {:2d} | Test_Loss  {:.10f} | Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Sens  {:.3f} | Test_Spec  {:.3f}\n'.format(
                    epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
                    )
            with open('log.txt', 'a') as F:
                F.writelines('Epoch {:2d} | Test_Loss  {:.10f} | Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Sens  {:.3f} | Test_Spec  {:.3f}\n'.format(
                epoch, loss_epoch, acc, auc, f1_score_, sens, spec))
        
        # save model
        if epoch % 10 == 0:
            torch.save(model, os.path.join(outdir,'DeepMOI_{}.pt'.format(epoch)))



