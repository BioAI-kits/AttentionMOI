import os, sys, time, torch, gzip
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, accuracy_score
from captum.attr import IntegratedGradients


def clin_process_tsi(in_file, out_file, threshold=2):
    """To process raw clinical dataset download from linkedomics, which is .tsi format.

    Args:
        in_file (string): The raw clinical dataset file. A .csv format file.
        threshold (int, optional): a threshold year using to split patients into LTS and Non-LTS. Defaults to 2.
        out_file (string): Output file name. A .csv format file
    Return:
        return .csv format file. 
    """
    df_clin = pd.read_table(in_file)
    # format the table
    df_clin = df_clin.T
    cols = df_clin.iloc[0,:]
    df_clin = df_clin.iloc[1:, :]
    df_clin.columns = cols
    # label the patients
    df_clin = df_clin[~df_clin.overall_survival.isna()]
    df_clin = df_clin[~df_clin.status.isna()]
    df_clin.overall_survival = df_clin.overall_survival.astype('int')
    df_clin.status = df_clin.status.astype('int')
    df_clin.loc[(df_clin.overall_survival < threshold*365) & (df_clin.status == 1), 'label'] = 1
    df_clin.loc[(df_clin.overall_survival >= threshold*365), 'label'] = 0
    df_clin = df_clin[~df_clin.label.isna()]
    # reset column names
    df_clin = df_clin.reset_index(drop=False)
    df_clin.columns = ['patient_id', 'feature_age', 'feature_tumor_purity', 'feature_histological_type',
                        'feature_gender', 'feature_radiation_therapy', 'feature_race', 'feature_ethnicity',
                        'overall_survival', 'status', 'overallsurvival', 'label']
    # output1:
    df_clin.to_csv(out_file + '.tmp.csv', index=False)
    
    # output2: One-hot encode
    df_clin = pd.read_csv(out_file + '.tmp.csv')
    cols = ['patient_id', 'label']
    for col in df_clin.columns.values:
        if col.startswith('feature'):
            cols.append(col)
    df_clin = df_clin[cols]
    df_clin = pd.concat([df_clin.iloc[:, :2],pd.get_dummies(df_clin.iloc[:, 2:])], axis=1)
    df_clin.to_csv(out_file, index=False)


def check_files(files):
    """To check files.
    files (str or list)
    """
    if isinstance (files,list):
        for f in files:
            if not os.path.exists(f):
                print('[Error] {} not found.'.format(f))
                sys.exit(1)
    elif isinstance (files,str):
        if not os.path.exists(files):
                print('[Error] {} not found.'.format(files))
                sys.exit(1)
    else:
        print('[Error] {} file path is wrong.'.format(files))
        sys.exit(1)


# init log.txt
def init_log(args):
    with open(os.path.join(args.outdir, 'log.txt'), 'w') as O:
        run_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        head_line = "Perform model training at {}".format(run_time)
        O.writelines(head_line + '\n\n')


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


# for explain
def ig(args, model, dataset, feature_names, omic_group, target=1):
    # prepare input data
    input_tensor_list = [d[0].unsqueeze(0) for d in dataset]
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
                           'Target': [target]*len(feature_names),
                           'Importance_value': feat_importance,
                           'Importance_value_abs': abs(feat_importance)
                           })
    df_imp = df_imp.sort_values('Importance_value_abs', ascending=False)

    # output 
    df_imp.to_csv(os.path.join(args.outdir, 'feature_importance_target{}.csv'.format(target)), index=False)
    
    return df_imp