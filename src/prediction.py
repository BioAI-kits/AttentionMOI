import os
import torch
import numpy as np
import joblib
import torch.nn.functional as F
from .utils import check_files
from .preprocess import read_dataset, read_
from .utils import evaluate


def read_feature(args):
    if args.FSD:
        feature_file = os.path.join(args.outdir, 'features_FSD_{}_{}.csv'.format(args.method, args.seed))
    else:
        feature_file = os.path.join(args.outdir, 'features_{}_{}.csv'.format(args.method, args.seed))
    print("Featrue file path is: "+feature_file)
    features = read_(feature_file)
    return features

def predict_DNN(args, data):
    if args.FSD:
        if args.clin_file:
            # to load model
            model_path =  os.path.join(args.outdir,'model_DNN_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed))
        else:
            # to load model
            model_path =  os.path.join(args.outdir,'model_DNN_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed))
    else:
        if args.clin_file:
            # to load model
            model_path =  os.path.join(args.outdir,'model_DNN_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed))
        else:
            # to load model
            model_path =  os.path.join(args.outdir,'model_DNN_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed))
    print("Model path is: "+model_path)
    model = torch.load(model_path)
    out = model(torch.FloatTensor(data))
    out = out.squeeze(-1)
    y_pred_prob = F.softmax(out).detach().tolist()
    return model, y_pred_prob


def predict_Net(args, data, features):
    if args.FSD:
        if args.clin_file:
            # to load model
            model_path =  os.path.join(args.outdir,'model_Net_FSD_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed))
        else:
            # to load model
            model_path =  os.path.join(args.outdir,'model_Net_FSD_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed))
    else:
        if args.clin_file:
            # to load model
            model_path =  os.path.join(args.outdir,'model_Net_{}_{}_clin_{}.pt'.format(args.method, args.omic_name, args.seed))
        else:
            # to load model
            model_path =  os.path.join(args.outdir,'model_Net_{}_{}_{}.pt'.format(args.method, args.omic_name, args.seed))
    print("Model path is: " + model_path)
    model = torch.load(model_path)
    idx_rna = []
    for i, v in enumerate(features["omic_group"]):
        if v == 'rna':
            idx_rna.append(i)

    idx_dna = []
    for i, v in enumerate(features["omic_group"]):
        if v == 'cnv' or v == 'met':
            idx_dna.append(i)

    data_dna = data[:, idx_dna]
    data_rna = data[:, idx_rna]
    out = model(torch.FloatTensor(data_dna), torch.FloatTensor(data_rna))
    out = out.squeeze(-1)
    y_pred_prob = F.softmax(out).detach().tolist()
    return model, y_pred_prob

def predict_ml(args, data, model_name):
    if args.FSD:
        if args.clin_file:
            # to load model
            model_path =  os.path.join(args.outdir,'model_{}_FSD_{}_{}_clin_{}.joblib'.format(model_name, args.method, args.omic_name, args.seed))
        else:
            # to load model
            model_path =  os.path.join(args.outdir,'model_{}_FSD_{}_{}_{}.joblib'.format(model_name, args.method, args.omic_name, args.seed))
    else:
        if args.clin_file:
            # to load model
            model_path =  os.path.join(args.outdir,'model_{}_{}_{}_clin_{}.joblib'.format(model_name, args.method, args.omic_name, args.seed))
        else:
            # to load model
            model_path =  os.path.join(args.outdir,'model_{}_{}_{}_{}.joblib'.format(model_name, args.method, args.omic_name, args.seed))
    print("Model path is: " + model_path)
    model = joblib.load(model_path)
    y_pred_prob = model.predict_proba(data)
    return model, y_pred_prob

def write_out(args, model_name, acc, prec, f1, auc, recall):
    with open(os.path.join(args.outdir, 'evaluation_external.txt'), 'a') as txt:
        if args.FSD:
            if args.clin_file:
                txt.writelines(
                    f"{model_name}\tFSD_{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(
                    f"{model_name}\tFSD_{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
        else:
            if args.clin_file:
                txt.writelines(
                    f"{model_name}\t{args.method}\t{args.omic_name} clin\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
            else:
                txt.writelines(
                    f"{model_name}\t{args.method}\t{args.omic_name}\t{acc}\t{prec}\t{f1}\t{auc}\t{recall}\n")
    txt.close()

def predict(args):
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # read external datasets
    df_omics, df_label, df_clin = read_dataset(args)
    labels = df_label.label.values

    # select features
    features = read_feature(args)
    chosen_omics = []

    for omic_num, df_omic in enumerate(df_omics):
        feat_name = features[features["omic_group"] == args.omic_name[omic_num]]["feat_name"]
        df_omic = df_omic[feat_name]
        chosen_omics.append(df_omic.values)
    data = np.concatenate(chosen_omics, 1)
    data = np.nan_to_num(data)
    print("The shape of external dataset is {}.".format(data.shape))

    if args.model == "DNN":
        model, y_pred_prob = predict_DNN(args, data)
        # # evaluate model
        acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_prob), real_label=np.array(labels))
        write_out(args, "DNN", acc, prec, f1, auc, recall)
    elif args.model in ["RF", "XGboost", "svm"]:
        model_name = args.model
        model, y_pred_prob = predict_ml(args, data, model_name)
        # # evaluate model
        acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_prob), real_label=np.array(labels))
        write_out(args, model_name, acc, prec, f1, auc, recall)
    elif args.model == "Net":
        model, y_pred_prob = predict_Net(args, data, features)
        # # evaluate model
        acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_prob), real_label=np.array(labels))
        write_out(args, "Met", acc, prec, f1, auc, recall)
    # multiple models
    elif args.model == "all":
        # DNN model
        model, y_pred_prob = predict_DNN(args, data)
        # # evaluate model
        acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_prob), real_label=np.array(labels))
        write_out(args, "DNN", acc, prec, f1, auc, recall)
        # Net model
        model, y_pred_prob = predict_Net(args, data, features)
        # # evaluate model
        acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_prob), real_label=np.array(labels))
        write_out(args, "Met", acc, prec, f1, auc, recall)
        # ml model
        for model_name in ["RF", "XGboost", "svm"]:
            model, y_pred_prob = predict_ml(args, data, model_name)
            # # evaluate model
            acc, prec, f1, auc, recall = evaluate(pred_prob=np.array(y_pred_prob), real_label=np.array(labels))
            write_out(args, model_name, acc, prec, f1, auc, recall)

