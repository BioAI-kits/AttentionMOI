import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from xgboost import XGBClassifier
from .module import DeepMOI
from .utils import evaluate


# builing machine learning models
def ml_models(args, data, chosen_feat_name, chosen_omic_group, labels, model_name):
    """
    args: using args - test_size, seed
    data: combined omics data (format:np.array)
    chosen_feat_name: feature names
    chosen_omic_group: feature of omic group
    labels: labels of classes
    return: print train and test ACC, AUC, Fi_score, Recall, Precision, output feature importance as a .csv file
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=args.test_size, random_state=args.seed)
    if args.model == "RF" or model_name == "RF":
        model = RandomForestClassifier(random_state=args.seed)
        model.fit(X_train, y_train)

    elif args.model == "XGboost" or model_name == "XGboost":
        model = XGBClassifier(random_state=args.seed)
        model.fit(X_train, y_train)

    elif args.model == "svm" or model_name == "svm":
        model = svm.SVC(random_state=args.seed, probability=True)
        model.fit(X_train, y_train)

    y_pred_train = model.predict_proba(X_train)
    y_pred_test = model.predict_proba(X_test)
    acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_train),
                                          real_label=np.array(y_train))
    print('Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
                acc, auc, f1, recall, prec))

    acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_test),
                                          real_label=np.array(y_test))
    test_evaluate = 'Test_ACC  {:.3f} | Test_AUC  {:.3f} | Test_F1_score  {:.3f} | Test_Recall  {:.3f} | Test_Precision  {:.3f}\n'.format(
                acc, auc, f1, recall, prec)
    print(test_evaluate)
    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        file.writelines("------Running {} model------\n".format(model_name))
        file.writelines(test_evaluate + '\n')
    file.close()

    # write output for model evaluation
    # model, feature selection, accuracy, precision, f1_score, AUC, recall
    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"{model_name}\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(f"{model_name}\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
        else:
            txt.writelines(f"{model_name}\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
    txt.close()

    if args.model == "svm" or model_name == "svm":
        importance = permutation_importance(model, X_train, y_train, scoring="accuracy")
        feature_imp = pd.DataFrame.from_dict({"features": chosen_feat_name,
                                              "omic_group": chosen_omic_group,
                                              "score": importance.importances_mean})
    else:
        feature_imp = pd.DataFrame.from_dict({"features": chosen_feat_name,
                                              "omic_group": chosen_omic_group,
                                              "score": model.feature_importances_})

    if args.FSD:
        if args.clin_file:
            feature_imp.to_csv(os.path.join(args.outdir, 'feature_imp_FDS_{}_{}_clin.csv'.format(args.method, model_name)), index=True)
        else:
            feature_imp.to_csv(os.path.join(args.outdir, 'feature_imp_FDS_{}_{}.csv'.format(args.method, model_name)), index=True)
    else:
        if args.clin_file:
            feature_imp.to_csv(os.path.join(args.outdir, 'feature_imp_{}_{}_clin.csv'.format(args.method, model_name)), index=True)
        else:
            feature_imp.to_csv(os.path.join(args.outdir, 'feature_imp_{}_{}.csv'.format(args.method, model_name)), index=True)

# using DNN model
def train(args, data, labels):
    # Convert data to tensor format
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i]), torch.tensor(labels[i])])
    # prepare
    in_dim = dataset[0][0].shape[0]  # input dim
    out_dim = len(set([int(d[1]) for d in dataset]))  # output dim
    model = DeepMOI(in_dim, out_dim)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dataset_train: object
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    loader = DataLoader(dataset_train, batch_size=args.batch)

    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        file.writelines("------Running DNN model------" + '\n')

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
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_train + '\n')
                file.writelines(log_test + '\n')

    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
        else:
            txt.writelines(f"DNN\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
    txt.close()

    file.close()
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


