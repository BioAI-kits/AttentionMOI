import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from .preprocess import read_dataset
from tqdm import tqdm
import torch
from .feature_selection import anova, rfe  # 导入Feature selection方法


def distribution_test_binaryclass(data, labels, sample_size=0.3, seed=0):
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


def distribution_test_multiclass(data, labels, sample_size=0.3, seed=0):
    # check label0 vs. label1: need (p<0.05)
    data_i = []
    for i in set(labels):
        idx = [True if n == i else False for n in labels]
        data_i.append(data[idx])

    _, pvalue_01 = stats.f_oneway(*data_i)

    # check train vs. test for total data: need (p>0.05)
    _, X_smp, _, Y_smp = train_test_split(
        data, labels, test_size=sample_size, random_state=seed)
    data_i = []
    for i in set(Y_smp):
        idx = [True if n == i else False for n in Y_smp]
        data_i.append(X_smp[idx])
    _, pvalue_02 = stats.f_oneway(*data_i)

    pvalue_03 = stats.f_oneway(X_smp, data).pvalue

    return pvalue_01, pvalue_02, pvalue_03


def feature_selection_distribution(args):
    # 1. to read input files
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
        # FSD to remove data noise
        print("Runing feature selection for {}\n".format(args.omic_name[omic_num]))
        choosed_feat_id = [] # save feature's id
        omic_data = df_omic.values
        for i in tqdm(range(omic_data.shape[1])):
            data = omic_data[:, i]
            m = 0
            for seed in seeds:
                # 二分类采用KS检验数据分布
                if len(set(labels)) == 2:
                    pvalue_01, pvalue_02, pvalue_03 = distribution_test_binaryclass(
                        data=data, labels=labels, seed=seed)
                # 多分类采用F检验数据分布
                else:
                    pvalue_01, pvalue_02, pvalue_03 = distribution_test_multiclass(
                        data=data, labels=labels, seed=seed)

                if pvalue_01 < 0.1 and pvalue_02 < 0.1 and pvalue_03 > 0.05:
                    m += 1
            m = m / len(seeds)
            if m > args.threshold:
                choosed_feat_id.append(i)
        
        # 如果没有特征通过FSD筛选
        if len(choosed_feat_id) == 0:
            continue
        df_omic = df_omic.iloc[:, choosed_feat_id]
        print("通过FSD筛选了 {} 个特征.".format(df_omic.shape[1]))

        ######################################################################
        #                     修改这个位置                                     #
        ######################################################################
        # Feature selection
        df_omic = anova(args=args, df_omic=df_omic, label=labels)
        print("通过特征提取筛选了 {} 个特征.".format(df_omic.shape[1]))

        # preprocessing result
        choosed_omics.append( df_omic.values )
        choosed_feat_name += df_omic.columns.to_list()  # Filtered feature name 
        choosed_omic_group += [args.omic_name[omic_num]] * len(df_omic.columns.to_list())  # The omics name corresponding to the filtered feature

    # 4. add clinical feature
    if not df_clin is None:
        choosed_omics.append(df_clin.values)
        choosed_feat_name += df_clin.columns.to_list()
        choosed_omic_group += ['Clin'] * df_clin.shape[1]

    # 5. merge omic dataset
    data = np.concatenate(choosed_omics, 1)
    data = np.nan_to_num(data)
    data = data.astype('float32')
    if data.shape[1] == 0:
        print('Program terminated! The reason is that no features passed the feature filter. Please check the data or adjust the feature selection parameters.')
        sys.exit(1)

    # 6. Convert data to tensor format
    dataset = []
    for i in range(len(labels)):
        dataset.append([torch.tensor(data[i]), torch.tensor(labels[i])])

    # 7. return dataset
    return dataset, choosed_feat_name, choosed_omic_group, labels
