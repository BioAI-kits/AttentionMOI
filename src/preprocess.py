import sys
import pandas as pd


def read_(file):
    # read file
    if file.endswith('.csv'):
        df = pd.read_csv(file, index_col=0)
    elif file.endswith('.csv.gz'):
        df = pd.read_csv(file, compression='gzip', index_col=0)
    else:
        print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
        sys.exit(0)
    return df


def read_omics(args):
    omics = []
    for file in args.omic_file:
        df = read_(file)
        df = df.fillna(0)  # fill nan with 0
        omics.append(df)
    return omics


def read_label(args):
    file = args.label_file
    df = read_(file)
    df = df.rename(
        columns={df.columns.values[0]: 'label'})
    return df


def read_clin(args):
    file = args.clin_file
    df = None
    if not file is None:
        df = read_(file)
        # fill na
        df = df.fillna(0)
    return df


def process(df_omics, df_label, df_clin):
    # extract patient id
    patients = [df_tmp.index.to_list() for df_tmp in df_omics]
    patients.append(df_label.index.to_list())
    if not df_clin is None:
        patients.append(df_clin.index.to_list())

    # get shared patients between different data
    patients_shared = patients[0]
    for i in range(1, len(patients)):
        patients_shared = list(set(patients_shared).intersection(patients[i]))

    # extract shared patients' data
    for i in range(len(df_omics)):
        df_omics[i] = df_omics[i].loc[patients_shared, :].sort_index()
    df_label = df_label.loc[patients_shared, :].sort_index()
    if not df_clin is None:
        df_clin = df_clin.loc[patients_shared, :].sort_index()
    return df_omics, df_label, df_clin


# api
def read_dataset(args):
    # 1. read raw dataset
    # (1) read omics dataset
    df_omics = read_omics(args)
    # (2) read label
    df_label = read_label(args)
    # (3) read clinical feature
    df_clin = read_clin(args)

    # 2. process
    df_omics, df_label, df_clin = process(df_omics, df_label, df_clin)

    # 3. return clean dataset
    return df_omics, df_label, df_clin
