import os, sys, torch, dgl
import numpy as np
import pandas as pd
from util import check_files

from torch_geometric.data import Data


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


def read_omics(omics_files, label_file, pathway_file, add_file=None):
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
        # read pathway file
        df_p = pd.read_csv(pathway_file)
        removed_pathways = []
        
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
            
            # del pathways whose overlapping gene number proportion < 0.8
            for pathway, group in df_p.groupby('pathway'):
                genes_ = set(group.src.to_list() + group.dest.to_list())
                omic_genes = set(df.gene.to_list())
                prop = len(omic_genes & genes_) / len(genes_)
                if prop < 0.8:
                    removed_pathways.append(pathway)
                    
            # fill miss gene for omic
            genes_ = set(df_p.src.to_list() + df_p.dest.to_list())
            df = df[df.gene.isin(genes_)]
            df = df.set_index('gene', drop=True)
            fill_idx = genes_ - set(df.index.to_list())
            for i in fill_idx:
                df.loc[i, :] = df.mean()
            df = df.reset_index(drop=False)
            
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

        # remove pathway
        pathway = df_p[~df_p.pathway.isin(removed_pathways)]

        # chang gene id to node id
        id_mapping = omics[0][['gene']].reset_index(drop=False)
        id_mapping.columns = ['node_id', 'gene_id']
        id_mapping = id_mapping.set_index('gene_id')
        pathway.src = pathway.src.map(lambda x: id_mapping.loc[x, 'node_id'])
        pathway.dest = pathway.dest.map(lambda x: id_mapping.loc[x, 'node_id'])

        # stack omic feature 
        omic_tensor = [torch.tensor(omic.iloc[:, 1:].values, dtype=torch.float32) for omic in omics]
        omic_tensor = torch.stack(omic_tensor, dim=2)

        # build graph
        G = Data(edge_index=torch.tensor([pathway.src.values, pathway.dest.values], dtype=torch.long),
                 x=omic_tensor
                )
        df_label = df_label[df_label.patient.isin(omics[0].columns.values[1:])]
        G.label= df_label.label.values


    return G, pathway, id_mapping



