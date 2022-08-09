import pandas as pd
import numpy as np
from scipy import stats
import sys
from sklearn.model_selection import train_test_split
import tqdm


# read omic files
def read_omics(args):
    omics = []
    for file in args.omic_file:
        # read omic file
        if file.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.endswith('.csv.gz'):
            df = pd.read_csv(file, compression='gzip')
        else:
            print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
            sys.exit(0)
        # rename first column
        df = df.rename(columns={df.columns.values[0]: 'gene'})
        # process
        # df = df.drop_duplicates('gene', keep='first')
        # df = df.sort_values('gene').reset_index(drop=True)
        df = df.fillna(0)  # fill nan with 0
        df = df.set_index('gene').T.sort_index()
        omics.append(df)
    return omics


# read label file
def read_label(args):
    file = args.label_file
    # read file
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith('.csv.gz'):
        df = pd.read_csv(file, compression='gzip')
    else:
        print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
        sys.exit(0)
    # rename first column
    df = df.rename(
        columns={df.columns.values[0]: 'patient', df.columns.values[1]: 'label'})
    return df


def process(omics, label):
    # keep shared patients between patients
    patients = []
    for omic in omics:
        patients.append(omic.index.values)
    patients.append(label.patient.values)
    patients_shared = patients[0]
    for i in range(1, len(patients)):
        patients_shared = list(set(patients_shared).intersection(patients[i]))
    # only keep shared patients
    omics_ = [omic.loc[patients_shared, :].sort_index() for omic in omics]
    label_ = label[label.patient.isin(patients_shared)].sort_values('patient')

    return omics_, label_


def distribution(data, labels, seed):
    # check label0 vs. label1
    label_1 = [True if i == 1 else False for i in labels]
    label_0 = [True if i == 0 else False for i in labels]
    pvalue_1 = stats.kstest(data[label_1], data[label_0]).pvalue

    # sample data
    _, X_sample, _, _ = train_test_split(
        data, labels, test_size=0.3, random_state=seed)
    pvalue_2 = stats.kstest(data, X_sample).pvalue

    return pvalue_1, pvalue_2


def check(omic, label, seeds, args, p1, p2):
    # single omic data matrix
    dat = omic.values
    # label
    labels = label.label.values

    # check by single feature
    candidates_list = []
    candidates_list.append(range(dat.shape[1]))
    for n, seed in enumerate(seeds):
        candidates_list.append([])
        for i in tqdm.tqdm(candidates_list[n]):
            data = dat[:, i]
            pvalue_01, pvalue_02 = distribution(
                data=data, labels=labels, seed=seed)
            if pvalue_01 < p1 and pvalue_02 > p2:
                candidates_list[n+1].append(i)
        print("Seed: {} | Candidates' Number: {}".format(
            seed, len(candidates_list[n+1])))

    return candidates_list[n+1]


# to select omics' features
def selection(args):
    # read dataset
    omics = read_omics(args)
    label = read_label(args)

    # process
    omics, label = process(omics, label)

    # selection using resample
    np.random.seed(args.seed)
    seeds = np.random.randint(0, 1000, args.iteration)
    omic_selection = []
    for a, omic in enumerate(omics):
        if a < 2:
            p1 = 0.01
            p2 = 0.05
        else:
            p1 = 0.05
            p2 = 0.05
        candidates = check(omic, label, seeds, args, p1, p2)
        omic_selection.append(omic.values[:, candidates])

    return omic_selection, label.label.values
