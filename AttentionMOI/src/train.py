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
import joblib
from .module import DeepMOI, Net
from .utils import evaluate
from .moanna.model.Moanna import Moanna, Moanna_cls
from .mogonet.main_mogonet import run_mogonet


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
        if args.FSD:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_FSD_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))
            else:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_FSD_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))

            else:
                joblib.dump(model, os.path.join(args.outdir,'model_RF_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))

    elif args.model == "XGboost" or model_name == "XGboost":
        model = XGBClassifier(random_state=args.seed)
        model.fit(X_train, y_train)
        if args.FSD:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_FSD_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))
            else:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_FSD_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))

            else:
                joblib.dump(model, os.path.join(args.outdir,'model_XGboost_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))

    elif args.model == "svm" or model_name == "svm":
        model = svm.SVC(random_state=args.seed, probability=True)
        model.fit(X_train, y_train)
        if args.FSD:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_FSD_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))
            else:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_FSD_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_{}_{}_clin_{}.joblib'.format(args.method, args.omic_name, args.seed)))

            else:
                joblib.dump(model, os.path.join(args.outdir,'model_svm_{}_{}_{}.joblib'.format(args.method, args.omic_name, args.seed)))

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
            if args.clin_file:
                txt.writelines(f"{model_name}\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(
                    f"{model_name}\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
    txt.close()


# using DNN model
def train(args, data, labels):
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    # in_dim, out_dim
    dim_out = len(set(labels))
    in_dim = data.shape[1]
    model = DeepMOI(in_dim, dim_out)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    train_loader = DataLoader(dataset_train, batch_size=args.batch)
    test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, drop_last=True)

    for epoch in range(args.epoch):
        # 1) training model
        model.train()
        loss_epoch = []
        y_pred_probs, real_labels = [], []
        for data, labels in train_loader:
            out = model(data)
            out = out.squeeze(-1)
            # loss
            loss = nn.CrossEntropyLoss()(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # prediction
            y_pred_prob = F.softmax(out).detach().tolist()
            y_pred_probs.append(y_pred_prob)
            real_labels.append(labels)
        loss_epoch = np.mean(loss_epoch)
        y_pred_probs = np.concatenate(y_pred_probs)
        real_labels = np.concatenate(real_labels)
        acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
        log_train = 'Epoch {:2d} | Train Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1, recall, prec)
        with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
            file.writelines(log_train + '\n')

        with torch.no_grad():
            # test metrics
            loss_epoch = []
            y_pred_probs, real_labels = [], []
            for data, labels in test_loader:
                out = model(data)
                out = out.squeeze(-1)
                # loss
                loss = nn.CrossEntropyLoss()(out, labels)
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                # predict labels
                y_pred_prob = F.softmax(out).detach().tolist()
                y_pred_probs.append(y_pred_prob)
                real_labels.append(labels)

            loss_epoch = np.mean(loss_epoch)
            y_pred_probs = np.concatenate(y_pred_probs)
            real_labels = np.concatenate(real_labels)
            acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
            log_test = 'Epoch {:2d} | Test Loss {:.10f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Recall {:.3f} | Test_Precision {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_test + '\n')
    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
    txt.close()

    return model, test_loader


def train_net(args, data, chosen_omic_group, labels):
    idx_rna = []
    for i, v in enumerate(chosen_omic_group):
        if v == 'rna':
            idx_rna.append(i)

    idx_dna = []
    for i, v in enumerate(chosen_omic_group):
        if v == 'cnv' or v == 'met':
            idx_dna.append(i)

    data_dna = data[:, idx_dna]
    data_rna = data[:, idx_rna]
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data_dna[i], dtype=torch.float),
                        torch.tensor(data_rna[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    # dim_dna, dim_rna, dim_out
    dim_dna = data_dna.shape[1]
    dim_rna = data_rna.shape[1]
    dim_out = len(set(labels))
    model = Net(dim_dna, dim_rna, dim_out)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    train_loader = DataLoader(dataset_train, batch_size=args.batch)
    test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, drop_last=True)

    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        file.writelines("------Running Net model------" + '\n')

    for epoch in range(args.epoch):
        # 1) training model
        model.train()
        loss_epoch = []
        y_pred_probs, real_labels = [], []
        for data_dna, data_rna, labels in train_loader:
            out = model(data_dna, data_rna)
            out = out.squeeze(-1)
            # loss
            loss = nn.CrossEntropyLoss()(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # prediction
            y_pred_prob = F.softmax(out).detach().tolist()
            y_pred_probs.append(y_pred_prob)
            real_labels.append(labels)
        loss_epoch = np.mean(loss_epoch)
        y_pred_probs = np.concatenate(y_pred_probs)
        real_labels = np.concatenate(real_labels)
        acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
        log_train = 'Epoch {:2d} | Train Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1, recall, prec)
        print(log_train)
        with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
            file.writelines(log_train + '\n')

        with torch.no_grad():
            # test metrics
            loss_epoch = []
            y_pred_probs, real_labels = [], []
            for data_dna, data_rna, labels in test_loader:
                out = model(data_dna, data_rna)
                out = out.squeeze(-1)
                # loss
                loss = nn.CrossEntropyLoss()(out, labels)
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                # predict labels
                y_pred_prob = F.softmax(out).detach().tolist()
                y_pred_probs.append(y_pred_prob)
                real_labels.append(labels)

            loss_epoch = np.mean(loss_epoch)
            y_pred_probs = np.concatenate(y_pred_probs)
            real_labels = np.concatenate(real_labels)
            acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
            log_test = 'Epoch {:2d} | Test Loss {:.10f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Recall {:.3f} | Test_Precision {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)

            # to write log info
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_test + '\n')

    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"Net\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"Net\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                txt.writelines(f"Net\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"Net\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_Net_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
    txt.close()


    return model, test_loader



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


# using Moanna model
def train_moanna(args, data, labels):    
     # Pre-defined hyperparameters
    params = {}
    params["input_size"] = data.shape[1] # Number of features
    params["n_layers"] = 1
    params["encoded_size"] = 64 # Number of Autoencoders neurons
    params["hidden_size"] = params["encoded_size"] * (2 ** (params["n_layers"] + 1))
    params["drop_prob"] = 0.5
    params["fnn_hidden_size"] = 40
    params["fnn_number_layers"] = 2
    params["num_classes"] = len(set(labels))
    params["fnn_num_epoch"] = 100
    params["fnn_learning_rate"] = 0.05

    # Setup model
    model = Moanna(
        params["input_size"], 
        params["hidden_size"], 
        params["encoded_size"], 
        params["n_layers"], 
        params["drop_prob"], 
        params["fnn_hidden_size"], 
        params["num_classes"], 
        params["fnn_number_layers"], 
        0.1,
    ) 
    
    model_cls = Moanna_cls(
        params["input_size"], 
        params["hidden_size"], 
        params["encoded_size"], 
        params["n_layers"], 
        params["drop_prob"], 
        params["fnn_hidden_size"], 
        params["num_classes"], 
        params["fnn_number_layers"], 
        0.1,
    ) 
    
    
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    # in_dim, out_dim
    dim_out = len(set(labels))
    in_dim = data.shape[1]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr, weight_decay=args.weight_decay)
    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    train_loader = DataLoader(dataset_train, batch_size=args.batch)
    test_loader = DataLoader(dataset_test, batch_size=args.batch, shuffle=True, drop_last=True)

    # 先train 编码器和解码器
    criterion1 = torch.nn.MSELoss(reduce=True)
    for _ in range(50):
        model.train()
        for data, labels in train_loader:
            encoded, decoded = model(data)
            loss = criterion1(decoded, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    for epoch in range(args.epoch):
        # 1) training model
        model_cls.train()
        loss_epoch = []
        y_pred_probs, real_labels = [], []
        for data, labels in train_loader:
            encoded, decoded = model(data)
            out = model_cls(encoded)
    #         out = out.squeeze(-1)
    #         # loss
            loss = nn.CrossEntropyLoss()(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch.append(loss.item())
            # prediction
            y_pred_prob = F.softmax(out).detach().tolist()
            y_pred_probs.append(y_pred_prob)
            real_labels.append(labels)
        loss_epoch = np.mean(loss_epoch)
        y_pred_probs = np.concatenate(y_pred_probs)
        real_labels = np.concatenate(real_labels)
        acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
        log_train = 'Epoch {:2d} | Train Loss {:.10f} | Train_ACC {:.3f} | Train_AUC {:.3f} | Train_F1_score {:.3f} | Train_Recall {:.3f} | Train_Precision {:.3f}'.format(
            epoch, loss_epoch, acc, auc, f1, recall, prec)
        print(log_train)
        with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
            file.writelines(log_train + '\n')

        with torch.no_grad():
            # test metrics
            loss_epoch = []
            y_pred_probs, real_labels = [], []
            for data, labels in test_loader:
                encoded, decoded = model(data)
                out = model_cls(encoded)
                # out = out.squeeze(-1)
                # loss
                loss = nn.CrossEntropyLoss()(out, labels)
                optimizer.zero_grad()
                loss_epoch.append(loss.item())
                # predict labels
                y_pred_prob = F.softmax(out).detach().tolist()
                y_pred_probs.append(y_pred_prob)
                real_labels.append(labels)

            loss_epoch = np.mean(loss_epoch)
            y_pred_probs = np.concatenate(y_pred_probs)
            real_labels = np.concatenate(real_labels)
            acc, prec, f1, auc, recall = evaluate(y_pred_probs, real_labels)
            log_test = 'Epoch {:2d} | Test Loss {:.10f} | Test_ACC {:.3f} | Test_AUC {:.3f} | Test_F1_score {:.3f} | Test_Recall {:.3f} | Test_Precision {:.3f}'.format(
                epoch, loss_epoch, acc, auc, f1, recall, prec)
            print(log_test)
            with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
                file.writelines(log_test + '\n')
    with open(os.path.join(args.outdir, 'evaluation.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
        else:
            if args.clin_file:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed)))
            else:
                txt.writelines(f"DNN\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
                # to save model
                torch.save(model, os.path.join(args.outdir,
                                               'model_DNN_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed)))
    txt.close()


# using MOGONET model
def train_mogonet(args, data, labels):
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i], dtype=torch.float),
                        torch.tensor(labels[i], dtype=torch.long)])

    dataset_train, dataset_test = train_test_split(
        dataset, test_size=args.test_size, random_state=args.seed)
    
    os.makedirs('tmp', exist_ok=True)
    dats, labs = [], []
    for dat, label in dataset_train:
        dats.append(dat.numpy())
        labs.append(label.numpy())
    a = np.vstack(dats)
    b = np.vstack(labs)
    np.savetxt('tmp/1_tr.csv', a, delimiter=',')
    np.savetxt('tmp/labels_tr.csv', b, delimiter=',')
    
    dats, labs = [], []
    for dat, label in dataset_test:
        dats.append(dat.numpy())
        labs.append(label.numpy())
    a = np.vstack(dats)
    b = np.vstack(labs)
    np.savetxt('tmp/1_te.csv', a, delimiter=',')
    np.savetxt('tmp/labels_te.csv', b, delimiter=',')
    
    
    logs = run_mogonet(num_epoch=args.epoch)
    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        for log in logs:
            file.writelines(log + '\n')
    