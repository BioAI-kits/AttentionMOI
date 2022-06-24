import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import dgl
from dgl.data import DGLDataset
from dgl.data.utils import download


def read_clin(file=None):
    """To read the clinical dataset.
    Args:
        file (str, optional): file name.
    Returns:
        pandas.DataFrame: 1st column is patient_id; 2nd column is label; other columns are clinical features.
    """
    if file == None:
        print('\nError: No clinical dataset input.')
        sys.exit(0)
    else:
        clin = pd.read_csv(file)
    return clin


def read_omics(omics_files=None, clin_file=None):
    """To read multi omics dataset. 
    Args:
        files (list): multi-omics file names
        id_mapping (str, optional): output the mapping relation of node_id and gene_id. Defaults to 'geneid_2_nodeid.csv'.
    Returns:
        omics: a list, including omics dataset
        geneid2nodeid: gene id to node id.
    """
    if not isinstance(omics_files, list):
        print('\nError: No omics dataset input.\n')
        sys.exit(0)
    elif len(omics_files) <= 1:
        print('\nError: Our algorithm only supports multi-omics.\n')
        sys.exit(0)
    else:
        raw_omics = []
        patients, genes = [], []
        for file in omics_files:
            df = pd.read_csv(file, compression='gzip')
            df = df.drop_duplicates('gene', keep='first')
            df = df.sort_values('gene').reset_index(drop=True)
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
        df_clin = read_clin(clin_file)
        patient_overlap = list(set(df_clin.patient_id.to_list()) & set(patient_overlap))  # overlap patients with clinical dataset
        
        # only keep overlap genes and patients
        omics = []
        for df in raw_omics:
            df = df[['gene'] + patient_overlap]
            df = df[df.gene.isin(gene_overlap)].sort_values('gene').reset_index(drop=True)
            omics.append(df)

    return omics
    

def read_pathways(id_mapping, file='./Pathway/pathway_genes.gmt'):
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


def build_graph(omics, clinical_file):
    """To build graph, using multi-omics as nodes' attributes; using ppi as graph.
    Args:
        omics (list): this list is from read_omics
        clinical_file (dataframe, pandas): patient_id | label | clinical_feature1 | clinical_feature2 | clinical_creature3 | ...
    Returns:
        G (list): graphs for each patient.
        labels (list): the label for each patient.
        clin_features (tensor): the clinical features.
    """ 
    # read ppi
    base_path = os.path.split(os.path.realpath(__file__))[0]
    ppi_1 = os.path.join(base_path, 'PPI', 'ppi_1.csv.gz')
    ppi_2 = os.path.join(base_path, 'PPI', 'ppi_2.csv.gz')
    ppi_3 = os.path.join(base_path, 'PPI', 'ppi_3.csv.gz')
    df_ppi = pd.concat([pd.read_csv(ppi_1, compression='gzip'),
                        pd.read_csv(ppi_2, compression='gzip'),
                        pd.read_csv(ppi_3, compression='gzip')
                    ]).reset_index(drop=True)
    
    # obtain overlapping genes between ppi and omics
    genes = set(df_ppi.src.to_list() + df_ppi.dest.to_list()) & set(omics[0].gene.to_list())  
    print('\n[INFO] The overlaping genes number between omics and ppi dataset is: {}'.format(len(genes)))
    
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
    
    # read clinical file
    df_clin = read_clin(clinical_file)
    df_clin = df_clin[df_clin.patient_id.isin(omics_[0].columns[1:].values)].reset_index(drop=True)
    clin_features = torch.tensor(df_clin.iloc[:, 2:].values).unsqueeze(2)  # clinical features, for i-th samples: clin_features[i]
    
    # multi-omics features
    omics_tensor = []
    for omic in omics_:
        omics_tensor.append(torch.tensor(omic.loc[:, df_clin.patient_id.values].values, dtype=torch.float32).unsqueeze(2))
    multi_omics = torch.stack(omics_tensor, 2).squeeze(3)
    
    # build graphs
    G = []  # graphs for each patient
    labels = []  # 
    for i in range(multi_omics.shape[1]):
        # step1: build graphs
        g = dgl.graph((df_ppi.src.to_list(), df_ppi.dest.to_list()))
        # add edge feature, scale to 0-1
        e = torch.tensor(df_ppi.score.values)
        e = e.reshape(-1,1)
        e = (e - e.min()) / (e.max() - e.min())
        g.edata['e'] = e
        # add node feature, scale to 0-1
        ndata = multi_omics[:,i,:]
        ndata = (ndata - ndata.min()) /(ndata.max() - ndata.min())
        g.ndata['h'] = ndata
        G.append(g)
        # step2: get labels
        label = df_clin.label.values[i]
        labels.append(label)
    labels = torch.tensor(labels, dtype=torch.long)

    return G, labels, clin_features, id_mapping
