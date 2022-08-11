import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from .preprocess import read_dataset
from tqdm import tqdm
import torch


def distribution_test(data, labels, sample_size=0.3, seed=0):
    # check label0 vs. label1: need (p<0.05)
    label_1 = [True if i == 1 else False for i in labels]
    label_0 = [True if i == 0 else False for i in labels]
    pvalue_01 = stats.kstest(data[label_1], data[label_0]).pvalue

    # check train vs. test for total data: need (p>0.05)
    _, X_smp, _, Y_smp = train_test_split(
        data, labels, test_size=sample_size, random_state=seed)
    label_1 = [True if i == 1 else False for i in Y_smp]
    label_0 = [True if i == 0 else False for i in Y_smp]
    pvalue_02 = stats.kstest(X_smp[label_1], X_smp[label_0]).pvalue
    pvalue_03 = stats.kstest(X_smp, data).pvalue

    return pvalue_01, pvalue_02, pvalue_03


def feature_selection(args):
    # 1. get clin dataset
    df_omics, df_label, df_clin = read_dataset(args)
    labels = df_label.label.values

    # 2. fix random seed
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 10000, args.iteration)

    # 3. selection: omic-wise selection
    choosed_omics = []  # save feature 
    choosed_feat_name = []  # save feature's name
    choosed_omic_group = []  # save feature's omic group
    for omic_num, df_omic in enumerate(df_omics):
        print("Runing feature selection for {}\n".format(args.omic_name[omic_num]))
        choosed_feat_id = []
        omic_data = df_omic.values
        for i in tqdm(range(omic_data.shape[1])):
            data = omic_data[:, i]
            m = 0
            for seed in seeds:
                pvalue_01, pvalue_02, pvalue_03 = distribution_test(
                    data=data, labels=labels, seed=seed)
                if pvalue_01 < 0.05 and pvalue_02 < 0.05 and pvalue_03 > 0.05:
                    m += 1
            m = m / len(seeds)
            if m > 0.7:
                choosed_feat_id.append(i)
                choosed_feat_name.append(df_omic.columns.values[i])
                choosed_omic_group.append(args.omic_name[omic_num])
        choosed_omics.append(
            df_omic.iloc[:, choosed_feat_id].values
        )

    # 4. merge omic dataset
    if not df_clin is None:
        choosed_omics.append(df_clin.values)
        choosed_feat_name += df_clin.columns.to_list()
        choosed_omic_group += ['Clin'] * df_clin.shape[1]

    data = np.concatenate(choosed_omics, 1)
    data = np.nan_to_num(data)
    data = data.astype('float32')

    # 5. data changed
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i]), torch.tensor(labels[i])])

    # 6. return dataset
    return dataset, choosed_feat_name, choosed_omic_group
