import numpy as np
import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from .module import DeepMOI
from .utils import evaluate


def train(args, dataset):
    # prepare
    in_dim = dataset[0][0].shape[0]  # input dim
    out_dim = len( set( [int(d[1]) for d in dataset] ) )  # output dim
    model = DeepMOI(in_dim, out_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    loader = DataLoader(dataset_train, batch_size=args.batch)

    # building model
    for epoch in range(args.epoch):
        # 1) training model
        model.train()
        loss_epoch = []
        for _, sample in enumerate(loader):
            X, Y = sample[0], sample[1]
            out = model(X)
            loss = nn.CrossEntropyLoss()(out, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
        loss_epoch = np.mean(loss_epoch)
        
        # 2) evaluation
        with torch.no_grad():
            # evaluation for training dataset
            acc, prec, f1, auc, recall = evaluation(model, dataset_train)
            log_train = 'Epoch {:2d} | Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
                    epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_train)

            # evaluation for testing dataset
            acc, prec, f1, auc, recall = evaluation(model, dataset_test)
            log_test = 'Epoch {:2d} | Loss {:.10f} | Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Recall  {:.3f} | Test_Precision  {:.3f}\n'.format(
                        epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)

            # to write log info
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as O:
                O.writelines(log_train + '\n')
                O.writelines(log_test + '\n')

    # to save model
    torch.save(model, os.path.join(args.outdir, 'model.pt'))
    
    # output
    return model, dataset_test


def evaluation(model, dataset):
    """
    To evaluate model performance for training dataset and testing dataset.

    model: taining model
    dataset: training dataset or testing dataset.
    """
    y_pred_probs, real_labels = [], []
    for d in dataset:
        out = model(d[0])
        y_pred_prob = F.softmax(out).detach().tolist()
        y_pred_probs.append(y_pred_prob)
        real_labels.append(d[1].tolist())
    acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_probs), 
                                          real_label=np.array(real_labels))
    return acc, prec, f1, auc, recall
