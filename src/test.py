import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tqdm
import numpy as np
import os, sys, torch, dgl
import numpy as np
import pandas as pd
from torch_geometric.data import Data

from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
import dgl.nn.pytorch as dglnn
# from dgl.nn import Set2Set

from torch_geometric.nn import SAGEConv, SAGPooling, Set2Set, GraphNorm, global_sort_pool, GlobalAttention
from torch_geometric.utils import add_self_loops, subgraph
# from torch_geometric.loader import DataLoader

from torch_geometric.data import Data
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import metrics

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


def distribution(data, labels, seed=0):
    # check label0 vs. label1: need (p<0.05)
    label_1 = [True if i==1 else False for i in labels]
    label_0 = [True if i==0 else False for i in labels]
    pvalue_01 = stats.kstest(data[label_1], data[label_0]).pvalue
    
    # check train vs. test for total data: need (p>0.05)
    _, X_test, _, y_test = train_test_split(data, labels, test_size=0.3, random_state=seed)
    
    pvalue_tt = stats.kstest(data, X_test).pvalue
#     print(data)
#     print(pvalue_01, pvalue_tt)
    return pvalue_01, pvalue_tt


class DeepMOI(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DeepMOI, self).__init__()
        self.lin1 = nn.Linear(in_feat, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
#         x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, 0.5)
        
        x = self.lin2(x)
        x = torch.relu(x)
#         x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, 0.5)
        
        x = self.lin3(x)
        x = torch.relu(x)
#         x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, 0.5)
        
#         x = nn.Dropout(p=0.4)(x)
        x = self.lin4(x)
        logit = torch.sigmoid(x)
        
        return logit



omics_files=["../dataset/GBM/GBM.expression.csv.gz", "../dataset/GBM/GBM.met.csv.gz", "../dataset/GBM/GBM.cnv.csv.gz"]
label_file = "../dataset/GBM/labels.csv"

df = pd.read_csv(omics_files[0], compression='gzip')
df = df.set_index('gene').T.sort_index()
dat = df.values
df = pd.read_csv(omics_files[0], compression='gzip')
df = df.set_index('gene').T.sort_index()

labels = pd.read_csv(label_file)
labels = labels.sort_values('patient_id')
labels = labels[labels.patient_id.isin(df.index.values)]
patients = labels.patient_id.values
labels = labels.label.values

# RNA
df = pd.read_csv(omics_files[0], compression='gzip')
df = df.rename(columns={df.columns.values[0] : 'gene'})
df = df.drop_duplicates('gene', keep='first')
df = df.sort_values('gene').reset_index(drop=True)
df = df.fillna(0)  # fill nan with 0
df = df.set_index('gene').T.sort_index()
df = df.loc[patients, :]
dat = df.values
df = pd.read_csv(omics_files[0], compression='gzip')
df = df.set_index('gene').T.sort_index()
df = df.loc[patients, :]
dat = df.values
dat_rna = dat
labels = pd.read_csv(label_file)
labels = labels.sort_values('patient_id')
labels = labels[labels.patient_id.isin(df.index.values)]
labels = labels.label.values

np.random.seed(42)
seeds = np.random.randint(0, 1000, 10)

candidates_list = []
candidates_list.append(range(dat.shape[1]))

for n, seed in enumerate(seeds):
    candidates_list.append([])
    for i in tqdm.tqdm(candidates_list[n]):
        data = dat[:, i]
#         print(data)
        pvalue_01, pvalue_tt = distribution(data=data, labels=labels, seed=seed)
        if pvalue_01 < 0.01 and pvalue_tt > 0.05:
            candidates_list[n+1].append(i)
    print("Seed: {} | Candidates' Number: {}".format(seed, len(candidates_list[n+1])))

candidates_rna = candidates_list

# Met
df = pd.read_csv(omics_files[1], compression='gzip')
df = df.set_index('gene').T.sort_index()
df = df.loc[patients, :]
dat = df.values
dat_met = dat
labels = pd.read_csv(label_file)
labels = labels.sort_values('patient_id')
labels = labels[labels.patient_id.isin(df.index.values)]
labels = labels.label.values

np.random.seed(42)
seeds = np.random.randint(0, 1000, 20)

candidates_list = []
candidates_list.append(range(dat.shape[1]))

for n, seed in enumerate(seeds):
    candidates_list.append([])
    for i in tqdm.tqdm(candidates_list[n]):
        data = dat[:, i]
        pvalue_01, pvalue_tt = distribution(data=data, labels=labels, seed=seed)
        if pvalue_01 < 0.01 and pvalue_tt > 0.05:
            candidates_list[n+1].append(i)
    print("Seed: {} | Candidates' Number: {}".format(seed, len(candidates_list[n+1])))

candidates_met = candidates_list

# CNV
df = pd.read_csv(omics_files[2], compression='gzip')
df = df.set_index('gene').T.sort_index()
df = df.loc[patients, :]
dat = df.values
dat_cnv = dat

np.random.seed(42)
seeds = np.random.randint(0, 1000, 20)

candidates_list = []
candidates_list.append(range(dat.shape[1]))

for n, seed in enumerate(seeds):
    candidates_list.append([])
    for i in tqdm.tqdm(candidates_list[n]):
        data = dat[:, i]
        pvalue_01, pvalue_tt = distribution(data=data, labels=labels, seed=seed)
        if pvalue_01 < 0.01 and pvalue_tt > 0.05:
            candidates_list[n+1].append(i)
    print("Seed: {} | Candidates' Number: {}".format(seed, len(candidates_list[n+1])))

candidates_cnv = candidates_list

### Model
data = np.concatenate([dat_rna[:, candidates_rna[-1]], dat_met[:, candidates_met[-1]], dat_cnv[:, candidates_cnv[-1]] ], 1)

data = np.nan_to_num(data)

data = data.astype('float32')

# data = torch.tensor([data, labels], dtype=torch.float32)
dataset = []
for i in range(len(labels)):
    dataset.append([torch.tensor(data[i]), torch.tensor(labels[i])])

indim= data.shape[1]
model = DeepMOI(indim, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)

dataset_train, dataset_test = train_test_split(dataset, test_size=0.3, random_state=seeds[-1]) 
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















