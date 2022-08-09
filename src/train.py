from .module import DeepMOI
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
from dgl.nn.pytorch.factory import KNNGraph


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
    TN, FP, FN, TP = confusion_matrix(y_true=real_labels, y_pred=pred).ravel()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(real_labels, logits, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # F1 score
    f1_score_ = f1_score(y_true=real_labels, y_pred=pred)
    # sens
    sens = TP/float(TP+FN)
    # spec
    spec = TN/float(TN+FP)
    return acc, auc, f1_score_, sens, spec


def evaluate_gnn(logits, real_labels):
    """
    logits: sigmoid
    real_labels (numpy.array, dim=1)
    
    Return
        acc, auc, f1_score_, sens, spec
    """
    # acc
    pred = [1 if i > 0.5 else 0 for i in logits]
    acc = np.sum(np.array(pred) == np.array(real_labels)) / len(real_labels)
    acc = np.sum(np.array(pred) == np.array(real_labels.reshape(1,-1))[0]) / len(real_labels)

    # matrix
    TN, FP, FN, TP = confusion_matrix(y_true=real_labels, y_pred=pred).ravel()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(real_labels, logits, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # F1 score
    f1_score_ = f1_score(y_true=real_labels, y_pred=pred)
    # sens / recall
    sens = TP/float(TP+FN)
    # spec
    spec = TN/float(TN+FP)
    return acc, auc, f1_score_, sens, spec


def train(dataset, labels, args):
    data = np.concatenate(dataset, 1)
    data = np.nan_to_num(data)
    data = data.astype('float32')


    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i,:]), torch.tensor(labels[i])])

    indim= data.shape[1]
    model = DeepMOI(indim, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
    dataset_train, dataset_test = train_test_split(dataset, test_size=0.3, random_state=args.seed)
    loader = DataLoader(dataset_train, batch_size=16)

    for epoch in range(500):
        model.train()
        loss_epoch = []
        for batch_ndx, sample in enumerate(loader):
            X = sample[0]
            Y = sample[1]
            logits = model(X)
            loss = nn.BCELoss()(logits, Y.reshape(-1,1).to(torch.float32))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            
        loss_epoch = np.mean(loss_epoch)
        
        y_train_proba, y_train = [], []
        for d in dataset_train:
            y_train_proba.append(model(d[0]).detach().numpy())
            y_train.append(d[1])
        acc, auc, f1_score_, sens, spec = evaluate(logits=y_train_proba, real_labels=y_train)
        print('Epoch {:2d} | Train_Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Sens {:.3f} | Train_Spec {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
                )
    
        y_test_proba, y_test = [], []
        for d in dataset_test:
            y_test_proba.append(model(d[0]).detach().numpy())
            y_test.append(d[1])
        acc, auc, f1_score_, sens, spec = evaluate(logits=y_test_proba, real_labels=y_test)
        print('Epoch {:2d} | Test_Loss  {:.10f} | Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Sens  {:.3f} | Test_Spec  {:.3f}\n'.format(
                    epoch, loss_epoch, acc, auc, f1_score_, sens, spec))


#     kg = KNNGraph(5)
#     sample_graph = kg(torch.tensor(data))
#     ll = torch.tensor(labels, dtype=torch.float32).reshape(-1,1)
#     x = torch.tensor(data)

#     sample_graph.ndata['label'] = torch.tensor(labels, dtype=torch.float32).reshape(-1,1)
#     sample_graph.ndata['x'] = torch.tensor(data)

#     sample_graph.ndata['x'].shape
#     train_idx, test_idx = data_split(labels=sample_graph.ndata['label'], test_size=0.3)
    
#     indim= data.shape[1]
#     model = DeepMOI(indim, 1)
#     opt = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

#     for epoch in range(1000):
#         model.train()
#         # 使用所有节点(全图)进行前向传播计算
#         logits = model(sample_graph, x)
        
#         # 计算损失值
#         loss = nn.BCELoss()(logits[train_idx], ll[train_idx] )
        
#         # 进行反向传播计算
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
        
#         loss_epoch = loss.item()
#         acc, auc, f1_score_, sens, spec = evaluate_gnn(logits=logits.detach().numpy()[train_idx], real_labels=sample_graph.ndata['label'][train_idx])
#         print('Epoch {:2d} | Train_Loss {:.3f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Sens {:.3f} | Train_Spec {:.3f}'.format(
#                 epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
#                 )

#         acc, auc, f1_score_, sens, spec = evaluate_gnn(logits=logits.detach().numpy()[test_idx], real_labels=sample_graph.ndata['label'][test_idx])
#         print('Epoch {:2d} | Test_Loss {:.3f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Sens {:.3f} | Test_Spec {:.3f}\n'.format(
#                 epoch, loss_epoch, acc, auc, f1_score_, sens, spec)
#                 )


# def data_split(labels, test_size):
#     """To split dataset into training dataset and testing dataset.
#     Args:
#         labels (numpy): The labels of samples.
#         test_size (float, 0-1): The proportion of test data.
    
#     Return:
#         (list) index of training data, index of testing data
#     """
#     test_number = int(len(labels) * test_size)
#     idx = list(range(len(labels)))
#     np.random.shuffle(idx)
#     return idx[test_number:], idx[:test_number]


