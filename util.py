import mygene
import gzip
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, f1_score
import numpy as np
import os
import sys

def pre_string(file, out_file='data.csv.gz'):
    """The function used to preprocess the file (9606.protein.links.full.v11.5.txt.gz) download from STRING database.

    Args:
        file (file name): the data file name.
        out_file: output file.
    """
    df = pd.read_table(file, compression='gzip', sep=' ')
    df.protein1 = df.protein1.map(lambda x: x.split('.')[1])
    df.protein2 = df.protein2.map(lambda x: x.split('.')[1])
    proteins = df.protein1.to_list() + df.protein2.to_list()
    proteins = list(set(proteins))
    mg = mygene.MyGeneInfo()
    df_tmp = mg.querymany(proteins, 
                          scopes='ensembl.protein', 
                          fields='entrezgene', 
                          species='human', 
                          as_dataframe=True)
    df['src'] = df.protein1.map(lambda x: df_tmp.loc[x, 'entrezgene'])
    df['dest'] = df.protein2.map(lambda x: df_tmp.loc[x, 'entrezgene'])
    df = df[(~df.src.isna()) & (~df.dest.isna())]
    df = df.loc[:, ['src', 'dest', 'combined_score']]
    df.columns = ['src', 'dest', 'score']
    df.to_csv(out_file, compression='gzip', index=False)
    

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


def evaluate(logits, real_labels):
    """
    logits (numpy.array, dim=2)
    real_labels (numpy.array, dim=1)
    
    Return
        acc, auc, f1_score_, sens, spec
    """
    # acc
    pred = logits.argmax(axis=1)
    acc = np.sum(pred == real_labels) / len(real_labels)
    # matrix
    TN, FP, FN, TP = confusion_matrix(y_true=real_labels, y_pred=pred).ravel()
    # auc
    fpr, tpr, thresholds = metrics.roc_curve(real_labels, logits[:,1], pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # F1 score
    f1_score_ = f1_score(y_true=real_labels, y_pred=pred)
    # sens
    sens = TP/float(TP+FN)
    # spec
    spec = TN/float(TN+FP)
    return acc, auc, f1_score_, sens, spec


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



