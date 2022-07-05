import os, sys, torch, dgl
import numpy as np
import pandas as pd
from util import check_files


def read_label(file=None):
    """To read label file. 
    Args:
        file (str): A .csv format file. 1st column <- sample id; 2nd column <- label.
    Returns:
        pandas.DataFrame
    """
    # check file
    file = file.strip()
    if file == None:
        print('\nError: No labels file input.')
        sys.exit(0)

    check_files(file)

    # read file
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith('.csv.gz'):
        df = pd.read_csv(file, compression='gzip')
    else:
        print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
        sys.exit(0)
    
    # rename first column
    df = df.rename(columns={df.columns.values[0] : 'patient', df.columns.values[1] : 'label'})

    return df


def read_additional_file(file=None):
    """To read label file. 
    Args:
        file (str): A .csv format file. 1st column <- sample id; other columns <- non-omic features.
    Returns:
        pandas.DataFrame
    """
    # check file
    file = file.strip()
    if file == None:
        print('\nError: No labels file input.')
        sys.exit(0)

    # read file
    check_files(file)

    # read file
    if file.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.endswith('.csv.gz'):
        df = pd.read_csv(file, compression='gzip')
    else:
        print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
        sys.exit(0)
    
    # rename first column
    df = df.rename(columns={df.columns.values[0] : 'patient'})

    return df


def read_omics(omics_files, label_file, add_file=None):
    """To read multi omics dataset. 
    Args:
        omics_files (list): multi-omics file names
        label_file (str): label file.
        add_file (optional, str): additional non-omic file.
    Returns:
        omics: a list, including omics dataset
    """
    if not isinstance(omics_files, list):
        print('\nError: No omics dataset input.\n')
        sys.exit(0)
    else:
        raw_omics = []
        patients, genes = [], []
        for file in omics_files:
            # read omic file
            if file.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.endswith('.csv.gz'):
                df = pd.read_csv(file, compression='gzip')
            else:
                print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported, please ensure that the file name suffix is .csv or .csv.gz.'.format(file))
                sys.exit(0)
            
            # rename first column
            df = df.rename(columns={df.columns.values[0] : 'gene'})

            # process
            df = df.drop_duplicates('gene', keep='first')
            df = df.sort_values('gene').reset_index(drop=True)
            df = df.fillna(0)  # fill nan with 0
            # df.iloc[:, 1:] = (df.iloc[:, 1:] - df.iloc[:, 1:].mean()) / df.iloc[:, 1:].std()  # normalization by z-score
            # df.iloc[:, 1:] = (df.iloc[:, 1:] - df.iloc[:, 1:].min()) / (df.iloc[:, 1:].max() - df.iloc[:, 1:].min())  # normalization by scale

            # post-process
            raw_omics.append(df)
            genes += df.gene.to_list()
            patients += df.columns.to_list()[1:]
            
        # to obtain overlap genes
        gene_count = pd.DataFrame(pd.value_counts(genes))
        gene_count = gene_count.reset_index(drop=False)
        gene_count.columns = ['gene', 'count_']
        gene_overlap = gene_count[gene_count.count_ == len(raw_omics)].gene.to_list()
        
        # to obtain overlap patients
        patient_count = pd.DataFrame(pd.value_counts(patients))
        patient_count = patient_count.reset_index(drop=False)
        patient_count.columns = ['patient', 'count_']
        patient_overlap = patient_count[patient_count.count_ == len(raw_omics)].patient.to_list()

        # overlap patients with label file
        df_label = read_label(label_file)
        patient_overlap = list(set(df_label.patient.to_list()) & set(patient_overlap))

        # overlap patients with additional feature
        if add_file != None:
            df_add = read_additional_file(add_file)
            patient_overlap = list(set(df_add.patient.to_list()) & set(patient_overlap))
               
        # only keep overlap genes and patients
        omics = []
        for df in raw_omics:
            df = df[['gene'] + patient_overlap]
            df = df[df.gene.isin(gene_overlap)].sort_values('gene').reset_index(drop=True)
            omics.append(df)

    return omics
    

def read_pathways(id_mapping, file):
    """To read pathway data (gmt format).
    id_mapping (pandas, dataframe): return from build_graph function. It is used to changed gene id to node id.
    file (str): pathways file.

    Return:
        pathways (dict): key <-- pathway name; value <-- nodes id, pandas array.
    """
    pathways = {}
    with open(file) as F:
        for line in F.readlines():
            line = line.strip().split("\t")
            genes = [int(i.strip()) for i in line[1:]]
            nodes_id = id_mapping.loc[set(genes) & set(id_mapping.index.values), 'node_id'].values
            pathways[line[0]] = nodes_id
    return pathways


def build_graph(omics, label_file, add_file=None, network_file='default'):
    """To build graph, using multi-omics as nodes' attributes; using ppi as graph.
    Args:
        omics (list): this list is from read_omics function.
        label_file (str): 
        add_file (str, optional): 
    Returns:
        g (dgl.graph): graph for patients. [nodes, patients, omics], dim1 <- gene node; dim2 <- patients; dim3 <- omic features
        labels (tensor): the label for each patient.
        add_features (tensor): non-omic features.
        id_mapping (data.frame) : changing gene id to node id.
    """ 
    # read ppi
    if network_file == 'default':
        base_path = os.path.split(os.path.realpath(__file__))[0]
        ppi_1 = os.path.join(base_path, 'PPI', 'ppi_1.csv.gz')
        ppi_2 = os.path.join(base_path, 'PPI', 'ppi_2.csv.gz')
        ppi_3 = os.path.join(base_path, 'PPI', 'ppi_3.csv.gz')
        df_ppi = pd.concat([pd.read_csv(ppi_1, compression='gzip'),
                            pd.read_csv(ppi_2, compression='gzip'),
                            pd.read_csv(ppi_3, compression='gzip')
                        ]).reset_index(drop=True)
    else:
        network_file = network_file.strip()
        if network_file.endswith('csv'):
            df_ppi = pd.read_csv(network_file)
        elif network_file.endswith('csv.gz'):
            df_ppi = pd.read_csv(network_file, compression='gzip')
        else:
            print('\n[Error]: The program cannot infer the format of {} . Currently, only the csv format is supported. Please ensure that the file name suffix is .csv or .csv.gz.'.format(network_file))
            sys.exit(0)
        # rename
        df_ppi = df_ppi.rename(columns={df_ppi.columns.values[0] : 'src',   
                                        df_ppi.columns.values[1] : 'dest'
                                        })
    
    # obtain overlapping genes between ppi and omics
    genes = set(df_ppi.src.to_list() + df_ppi.dest.to_list()) & set(omics[0].gene.to_list())  
    print('[INFO] The overlaping genes number between omics and ppi dataset is: {}\n'.format(len(genes)))
    
    # only keep overlapping genes for ppi
    df_ppi = df_ppi[(df_ppi.src.isin(genes)) & ((df_ppi.dest.isin(genes)))].reset_index(drop=True)
    
    # only keep overlapping genes for omic
    omics = [omic[omic.gene.isin(genes)].reset_index(drop=True) for omic in omics]
    
    # to construct id mapping dataframe
    omic = omics[0]
    id_mapping = omic[omic.gene.isin(genes)].reset_index(drop=True)[['gene']].reset_index(drop=False)
    id_mapping.columns = ['node_id', 'gene_id']
    id_mapping = id_mapping.set_index('gene_id', drop=True)
    
    # change gene_id to node_id for ppi
    df_ppi.src = df_ppi.src.map(lambda x: id_mapping.loc[x, 'node_id'])
    df_ppi.dest = df_ppi.dest.map(lambda x: id_mapping.loc[x, 'node_id'])
    
    # change gene_id to node_id for omics
    omics_ = []
    for omic in omics:
        omic.gene = omic.gene.map(lambda x: id_mapping.loc[x, 'node_id'])
        omic = omic.sort_values('gene').reset_index(drop=True)
        omics_.append(omic)
    
    # read label
    df_label = read_label(label_file)
    df_label = df_label[df_label.patient.isin(omics_[0].columns[1:].values)].reset_index(drop=True)

    # read additional features
    if add_file != None:
        df_add = read_additional_file(add_file)
        df_add = df_add[df_add.patient.isin(omics_[0].columns[1:].values)].reset_index(drop=True)
        # df_add.iloc[:, 1:] = (df_add.iloc[:, 1:] - df_add.iloc[:, 1:].mean()) / df_add.iloc[:, 1:].std()  # normalization by z-score
        # df_add.iloc[:, 1:] = (df_add.iloc[:, 1:] - df_add.iloc[:, 1:].min()) / (df_add.iloc[:, 1:].max() - df_add.iloc[:, 1:].min())  # normalization by scale
        add_features = torch.tensor(df_add.iloc[:, 1:].values, dtype=torch.float32).unsqueeze(2)  # clinical features, for i-th samples: clin_features[i]
        add_features = torch.where(add_features.isnan(), torch.full_like(add_features, 0), add_features)  # fill nan with 0
    else:
        add_features = None

    # multi-omics features
    omics_tensor = []
    for omic in omics_:
        omic_data = torch.tensor(omic.loc[:, df_label.patient.values].values, dtype=torch.float32).unsqueeze(2)
        omic_data = torch.where(omic_data.isnan(), torch.full_like(omic_data, 0), omic_data)  # fill nan with 0
        omics_tensor.append(omic_data)
    multi_omics = torch.stack(omics_tensor, 2).squeeze(3)
    
    # build graph
    g = dgl.graph((df_ppi.src.to_list(), df_ppi.dest.to_list()))
    # add edge feature, scale to 0-1
    e = torch.tensor(df_ppi.score.values)
    e = e.reshape(-1,1)
    e = (e - e.min()) / (e.max() - e.min())
    g.edata['e'] = e
    # add node features
    g.ndata['h'] = multi_omics

    labels = df_label.label.values

    return g, labels, add_features, id_mapping


if __name__ == "__main__":
    omics = read_omics(omics_files=["./data/GBM/GBM.cnv.csv.gz", "./data/GBM/GBM.met.csv.gz"], label_file="./data/GBM/labels.csv", add_file="./data/GBM/clinincal.csv")
    g, labels, add_features, id_mapping = build_graph(omics=omics, label_file="./data/GBM/labels.csv", add_file="./data/GBM/clinincal.csv")
