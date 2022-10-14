import os, sys, time, torch, gzip
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, accuracy_score
from captum.attr import IntegratedGradients
from sklearn.impute import KNNImputer


def clin_read_tsi(in_file, out_file, task, threshold=2):
    """To process raw clinical dataset download from linkedomics, which is .tsi format.

    Args:
        in_file (string): The raw clinical dataset file, end with .tsi, a tab delinimated file.
        out_file (string): Output file name. A .csv format file
        task(string): Define task type. Could be one of the following tasks: LST, pan-class
        threshold (int, optional): a threshold year using to split patients into LTS and Non-LTS. Defaults to 2.
    Return:
        return .csv format file. 
    """
    df_clin = pd.read_table(in_file, index_col=0)
    # format the table
    df_clin = df_clin.T
    df_clin = df_clin[~df_clin.overall_survival.isna()]
    df_clin = df_clin[~df_clin.status.isna()]
    df_clin.overall_survival = df_clin.overall_survival.astype('int')
    df_clin.status = df_clin.status.astype('int')
    if task=="LST":
        # label patients
        df_clin.loc[(df_clin.overall_survival < threshold*365) & (df_clin.status == 1), 'label'] = 1
        df_clin.loc[(df_clin.overall_survival >= threshold*365), 'label'] = 0
        df_clin = df_clin[~df_clin.label.isna()]
    else:
        # label patients
        value_map = dict((v, i) for i, v in enumerate(pd.unique(df_clin["histological_type"])))
        print("\nLabels are encoded in to {} categories, encoding dictionary is: {}".format(len(value_map), value_map))
        df_clin["label"] = df_clin["histological_type"]
        df_clin = df_clin.replace({"label":value_map})
    # output: clinical file and label file
    df_clin.to_csv(out_file + "_clinical.csv", index=True)
    label = df_clin[["label"]].astype("int64")
    label.to_csv(out_file + "_label.csv", index=True)


# clinical feature processing
# add mutation info to clinical feature - CGC genes and total mutated gene number
# one hot encode of features
def process_clin(df_clin, df_mut, outfile, task):
    df_mut = df_mut.fillna(0)
    df = df_clin.merge(df_mut, left_index=True, right_index=True, how="left")
    if task == "LST":
        df = df.drop(["overall_survival", "status", "overallsurvival", "label"], axis=1)
        categorical = df.columns[(df.dtypes == "object").values].to_list()
        df = pd.concat([df.drop(categorical, axis=1),
                        pd.get_dummies(df[categorical])],
                       axis=1)
    else:
        df = df.drop(["histological_type", "label", "overallsurvival"], axis=1)
        categorical = df.columns[(df.dtypes == "object").values].to_list()
        df = pd.concat([df.drop(categorical, axis=1),
                        pd.get_dummies(df[categorical])],
                       axis=1)
    df.to_csv(outfile + "_clinical_tmp.csv", index=True)


def check_files(files):
    """To check files.
    files (str or list)
    """
    if isinstance(files, list):
        for f in files:
            if not os.path.exists(f):
                print('[Error] {} not found.'.format(f))
                sys.exit(1)
    elif isinstance(files, str):
        if not os.path.exists(files):
            print('[Error] {} not found.'.format(files))
            sys.exit(1)
    else:
        print('[Error] {} file path is wrong.'.format(files))
        sys.exit(1)


# init log.txt
def init_log(args):
    with open(os.path.join(args.outdir, 'log.txt'), 'a') as file:
        run_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        head_line = "Perform model training at {}".format(run_time)
        file.writelines(head_line + '\n\n')
        if args.FSD:
            if args.clin_file:
                file.writelines("-------Using FSD and clinical file, feature selection method {}, omic group {} clin, and model {}------------\n".format(args.method, args.omic_name, args.model))
            else:
                file.writelines("-------Using FSD, feature selection method {}, omic group {}, and model {}------------\n".format(args.method, args.omic_name, args.model))
        else:
            file.writelines("-------No FSD, using feature selection method {}, omic group {}, and model {}------------\n".format(args.method, args.omic_name, args.model))
    file.close()

# for evaluation
def evaluate(pred_prob, real_label, average="macro"):
    # For evaluating binary classification models
    if pred_prob.shape[1] == 2:
        y_pred = np.argmax(pred_prob, 1)
        prec = precision_score(real_label, y_pred)
        acc = accuracy_score(real_label, y_pred)
        f1 = f1_score(real_label, y_pred)
        recall = recall_score(real_label, y_pred)
        auc = roc_auc_score(real_label, pred_prob[:, 1])
    # For evaluating multiclass models
    else:
        y_pred = np.argmax(pred_prob, 1)
        prec = precision_score(real_label, y_pred, average='macro')
        acc = accuracy_score(real_label, y_pred)
        f1 = f1_score(real_label, y_pred, average=average)
        recall = recall_score(real_label, y_pred, average=average)
        auc = roc_auc_score(real_label, pred_prob, average='macro', multi_class='ovo')
    return acc, prec, f1, auc, recall


# for explain dnn model
def ig(args, model, dataset, feature_names, omic_group, target=1):
    # prepare input data
    input_tensor_list = [data for data, labels in dataset]
    input_tensor = torch.cat(input_tensor_list, 0)
    input_tensor.requires_grad_()

    # instantiation
    ig = IntegratedGradients(model)

    # calculating feature importance using IG
    attr, _ = ig.attribute(input_tensor, return_convergence_delta=True, target=target)
    attr = attr.detach().numpy()
    feat_importance = np.mean(attr, axis=0)

    # result
    df_imp = pd.DataFrame({'Feature': feature_names,
                           'Omic': omic_group,
                           'Target': [target] * len(feature_names),
                           'Importance_value': feat_importance,
                           'Importance_value_abs': abs(feat_importance)
                           })
    df_imp = df_imp.sort_values('Importance_value_abs', ascending=False)

    # output
    if args.FSD:
        if args.clin_file:
            df_imp.to_csv(os.path.join(args.outdir, 'feature_importance_FSD_{}_clin_DNN_{}_target{}.csv'.format(args.method, args.omic_name, target)), index=False)
        else:
            df_imp.to_csv(os.path.join(args.outdir, 'feature_importance_FSD_{}_DNN_{}_target{}.csv'.format(args.method, args.omic_name, target)),index=False)
    else:
        if args.clin_file:
            df_imp.to_csv(os.path.join(args.outdir, 'feature_importance_{}_clin_DNN_{}_target{}.csv'.format(args.method, args.omic_name, target)), index=False)
        else:
            df_imp.to_csv(os.path.join(args.outdir, 'feature_importance_{}_DNN_{}_target{}.csv'.format(args.method, args.omic_name, target)),index=False)

    return df_imp

 # for explain net model
def ig_net(args, model, dataset, feature_names, omic_group, target=1):
    # prepare input data
    input_tensor_dna, input_tensor_rna = [], []
    for data_dna, data_rna, labels in dataset:
        input_tensor_dna.append(data_dna)
        input_tensor_rna.append(data_rna)
    input_tensor_dna = torch.cat(input_tensor_dna, 0).requires_grad_()
    input_tensor_rna = torch.cat(input_tensor_rna, 0).requires_grad_()

    # instantiation
    ig = IntegratedGradients(model)

    # calculating feature importance using IG
    attr, _ = ig.attribute((input_tensor_dna, input_tensor_rna), return_convergence_delta=True, target=target)
    feat_importance = []
    for tensor in attr:
        tensor = tensor.detach().numpy()
        feat_importance.append(np.mean(tensor, axis=0))

    # result
    df_imp = pd.DataFrame({'Feature': feature_names,
                           'Omic': omic_group,
                           'Target': [target] * len(feature_names),
                           'Importance_value': np.concatenate(feat_importance),
                           'Importance_value_abs':  abs(np.concatenate(feat_importance))
                           })
    df_imp = df_imp.sort_values('Importance_value_abs', ascending=False)

    # output
    if args.FSD:
        if args.clin_file:
            df_imp.to_csv(
                os.path.join(args.outdir, 'feature_importance_FSD_{}_clin_Net_{}_target{}.csv'.format(args.method, args.omic_name, target)),
                index=False)
        else:
            df_imp.to_csv(
                os.path.join(args.outdir, 'feature_importance_FSD_{}_Net_{}_target{}.csv'.format(args.method, args.omic_name, target)),
                index=False)
    else:
        if args.clin_file:
            df_imp.to_csv(
                os.path.join(args.outdir, 'feature_importance_{}_clin_Net_{}_target{}.csv'.format(args.method, args.omic_name, target)),
                index=False)
        else:
            df_imp.to_csv(os.path.join(args.outdir, 'feature_importance_{}_Net_{}_target{}.csv'.format(args.method, args.omic_name, target)),
                          index=False)
    return df_imp