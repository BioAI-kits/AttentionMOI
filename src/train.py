from .module import DeepMOI
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np


def evaluate(logits, real_labels):
    """
    logits: sigmoid
    real_labels (numpy.array, dim=1)
    
    Return
        acc, auc, f1_score_, sens, spec
    """
    # acc
    pred = [1 if i > 0.5 else 0 for i in logits]
    acc = np.sum(np.array(pred) == np.array(real_labels)) / len(real_labels)
    # matrix
    TN, FP, FN, TP = metrics.confusion_matrix(y_true=real_labels, y_pred=pred).ravel()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(real_labels, logits, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # F1 score
    f1_score_ = metrics.f1_score(y_true=real_labels, y_pred=pred)
    # sens
    sens = TP/float(TP+FN)
    # spec
    spec = TN/float(TN+FP)
    return acc, auc, f1_score_, sens, spec


def train(args, dataset):
    # prepare
    in_dim = dataset[0][0].shape[0]  # input dim
    model = DeepMOI(in_dim, 1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=0.002)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    loader = DataLoader(dataset_train, batch_size=args.batch)

    # training model
    for epoch in range(args.epoch):
        model.train()
        loss_epoch = []
        for _, sample in enumerate(loader):
            X = sample[0]
            Y = sample[1]
            logits = model(X)
            loss = nn.BCELoss()(logits, Y.reshape(-1,1).to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            
        loss_epoch = np.mean(loss_epoch)
        
        # evaluation for training dataset
        y_train_proba, y_train = [], []
        for d in dataset_train:
            y_train_proba.append(model(d[0]).detach().numpy())
            y_train.append(d[1])
        acc, auc, f1_score_, sens, spec = evaluate(logits=y_train_proba, real_labels=y_train)
        print('Epoch {:2d} | Train_Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Sens {:.3f} | Train_Spec {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
                )
        # evaluation for testing dataset
        y_test_proba, y_test = [], []
        for d in dataset_test:
            y_test_proba.append(model(d[0]).detach().numpy())
            y_test.append(d[1])
        acc, auc, f1_score_, sens, spec = evaluate(logits=y_test_proba, real_labels=y_test)
        print('Epoch {:2d} | Test_Loss  {:.10f} | Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Sens  {:.3f} | Test_Spec  {:.3f}\n'.format(
                    epoch, loss_epoch, acc, auc, f1_score_, sens, spec))
    
    # output
    return model, dataset_test