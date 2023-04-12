def rename_feature(df, suffix):
    new_df_index = [x + suffix for x in df.index.tolist()]
    new_df_index_dict = {}
    for x in new_df_index:
        new_df_index_dict[x.split("_")[0]] = x
    df.rename(index=new_df_index_dict, inplace=True)

def standardise(df):
    df_transpose = df.transpose()
    df_transpose_rescale = (df_transpose - df_transpose.mean()) / df_transpose.std()
    return (df_transpose_rescale.transpose())

def print_shape(df, message, datatype):
    print(message, f'[{datatype}]: ', df.shape)

def print_mult_data_shape(expr_df, cnv_df, snp_df, message):
    print_shape(expr_df, message, "EXPR")
    print_shape(cnv_df, message, "CNV")
    print_shape(snp_df, message, "SNP")

def preprocess_tcga_df(expr_df, cnv_df, snp_df):
    print_mult_data_shape(expr_df, cnv_df, snp_df, "Original")
    
    # Re-header
    expr_df.set_index("Unnamed: 0", inplace=True)
    cnv_df.set_index("GeneName", inplace=True)
    snp_df.set_index("Unnamed: 0", inplace=True)
    
    # Replace "-" with "."
    cnv_df_list = list(cnv_df.columns.values)
    cnv_df_colnames = { x:x.replace("-",".") for x in cnv_df_list}
    cnv_df_tmp = cnv_df.rename(cnv_df_colnames, axis='columns')
    
    # Get the intersection of samplenames
    expr_df_list = list(expr_df.columns.values)
    cnv_df_list = list(cnv_df_tmp.columns.values)
    snp_df_list = list(snp_df.columns.values)

    samplenames = sorted(list(set(expr_df_list) & set(cnv_df_list) & set(snp_df_list)))

    expr_df_intersect = expr_df[expr_df.columns.intersection(samplenames)]
    cnv_df_intersect = cnv_df_tmp[cnv_df_tmp.columns.intersection(samplenames)]
    snp_df_intersect = snp_df[snp_df.columns.intersection(samplenames)]

    # Remove some rows from CNV, EXPR, SNP
    expr_df_rownames = expr_df_intersect.index.tolist()
    cnv_df_rownames = cnv_df_intersect.index.tolist()

    genenames = sorted(list(set(expr_df_rownames) & set(cnv_df_rownames)))

    cnv_df_final = cnv_df_intersect.loc[cnv_df_intersect.index.isin(genenames)]
    snp_df_final = snp_df_intersect
    expr_df_final = expr_df_intersect.loc[expr_df_intersect.index.isin(genenames)]
    expr_df_final.dropna(inplace=True)
    
    rename_feature(expr_df_final, "_EXPR")
    rename_feature(cnv_df_final, "_CNV")
    rename_feature(snp_df_final, "_SNP")
    
    print_mult_data_shape(expr_df_final, cnv_df_final, snp_df_final, "PreProcess")
    
    return expr_df_final, cnv_df_final, snp_df_final

def preprocess_metabric_df(expr_df, cnv_df, snp_df):
    print_mult_data_shape(expr_df, cnv_df, snp_df, "Original")
    
    # Re-header
    expr_df.set_index("Unnamed: 0", inplace=True)
    cnv_df.set_index("GeneName", inplace=True)
    snp_df.set_index("GeneName", inplace=True)
    
    # Replace "-" with "_"
    cnv_df_list = list(cnv_df.columns.values)
    cnv_df_colnames = { x:x.replace("-","_") for x in cnv_df_list}
    cnv_df_tmp = cnv_df.rename(cnv_df_colnames, axis='columns')
    
    # Replace "-" with "_"
    snp_df_list = list(snp_df.columns.values)
    snp_df_colnames = { x:x.replace("-","_") for x in snp_df_list}
    snp_df_tmp = snp_df.rename(snp_df_colnames, axis='columns')
    
    # Get the intersection of samplenames
    expr_df_list = list(expr_df.columns.values)
    cnv_df_list = list(cnv_df_tmp.columns.values)
    snp_df_list = list(snp_df_tmp.columns.values)

    samplenames = sorted(list(set(expr_df_list) & set(cnv_df_list) & set(snp_df_list)))

    expr_df_intersect = expr_df[expr_df.columns.intersection(samplenames)]
    cnv_df_intersect = cnv_df_tmp[cnv_df_tmp.columns.intersection(samplenames)]
    snp_df_intersect = snp_df_tmp[snp_df_tmp.columns.intersection(samplenames)]

    # Remove some rows from CNV
    expr_df_rownames = expr_df_intersect.index.tolist()
    cnv_df_rownames = cnv_df_intersect.index.tolist()

    genenames = sorted(list(set(expr_df_rownames) & set(cnv_df_rownames)))

    cnv_df_final = cnv_df_intersect.loc[cnv_df_intersect.index.isin(genenames)]
    snp_df_final = snp_df_intersect
    expr_df_final = expr_df_intersect.loc[expr_df_intersect.index.isin(genenames)]
    expr_df_final.dropna(inplace=True)
    
    rename_feature(expr_df_final, "_EXPR")
    rename_feature(cnv_df_final, "_CNV")
    rename_feature(snp_df_final, "_SNP")
    
    print_mult_data_shape(expr_df_final, cnv_df_final, snp_df_final, "PreProcess")
    
    return expr_df_final, cnv_df_final, snp_df_final

def preprocess_tcga_clin_df(clin_df_orig, expr_df):
    clin_df = clin_df_orig.copy()
    print_shape(clin_df, "Original", "ClinData")
    
    clin_df.set_index("SampleName", inplace=True)
    samplenames = clin_df.index.tolist()
    samplenames_expr = expr_df.columns.values
    samplenames_intersection = list(set(samplenames) & set(samplenames_expr))
    clin_df_final = clin_df.loc[clin_df.index.isin(samplenames_intersection)]
    clin_df_final = clin_df_final.reindex(expr_df.transpose().index)
                                    
    print_shape(clin_df_final, "PreProcess", "ClinData")
    
    return clin_df_final

def filter_clin_df(clin_df_orig, column, value):
    clin_df = clin_df_orig.copy()
    print_shape(clin_df, "Original", "ClinData")    
    
    # Filter
    clin_df_filtered = clin_df.loc[clin_df[column] == value]
    print_shape(clin_df_filtered, "Filtered", "ClinData")
    
    return clin_df_filtered

def add_tils_category(dat_df):
    import numpy as np
    
    new_dat_df = dat_df.copy()
    #Edit TILs info
    new_dat_df.loc[new_dat_df['tils_HandE_slides'].isna(),"TilsCategory"] = -1 # Will be replaced with NA at the last line
    new_dat_df.loc[(new_dat_df['tils_HandE_slides'] >= 0) & (new_dat_df['tils_HandE_slides'] < 30),"TilsCategory"] = 0
    new_dat_df.loc[new_dat_df['tils_HandE_slides'] >= 30,"TilsCategory"] = 1
    new_dat_df.loc[new_dat_df['tils_HandE_slides'] < 0,"TilsCategory"] = np.NaN
    
    return new_dat_df

def my_ttest(gene, group1, group2):
    import researchpy as rp
    descriptives, results = rp.ttest(group1[gene],group2[gene])
    ttest_pval = results.iloc[[3],[1]].values[0][0]
    
    return ttest_pval


def datasets_split(expr_df, cnv_df, snp_df, clin_df, split, method):
    if method == "random":
        split_method = split_samples_random
    
    train_samples, test_samples = split_method(clin_df, split)
    
    expr_df_train = expr_df[train_samples]
    expr_df_test = expr_df[test_samples]
    
    cnv_df_train = cnv_df[train_samples]
    cnv_df_test = cnv_df[test_samples]
    
    snp_df_train = snp_df[train_samples]
    snp_df_test = snp_df[test_samples]
    
    clin_df_train = clin_df.loc[train_samples]
    clin_df_test = clin_df.loc[test_samples]
    
    return expr_df_train, expr_df_test, cnv_df_train, cnv_df_test, snp_df_train, snp_df_test, clin_df_train, clin_df_test

def prep_data_label_remove_undefined(dat_tensor, label_df, label_type, labels_mapping):
    import pandas as pd
    import torch 
    import numpy as np
    
    # Get rid of Undefined
    if label_type == "Pam50Subtype":
        NAs = "Undefined"
    else:    
        NAs = "NA"
        
    undefined_list = label_df.index[label_df[label_type] == NAs].tolist()
    undefined_list += label_df.index[label_df[label_type].isna()].tolist()
    dat_df = pd.DataFrame(data = dat_tensor.detach().numpy(), index=label_df.index)
    
    dat_filtered_df = dat_df.loc[~dat_df.index.isin(undefined_list)]
    label_filtered_df = label_df.loc[~label_df.index.isin(undefined_list)].replace(labels_mapping)
    
    # convert to pytorch tensor
    dat_tensor = torch.from_numpy(dat_filtered_df.values).float()
    label_tensor = torch.from_numpy(label_filtered_df[label_type].values.astype(np.int32)).long().view(-1)
    
    return dat_tensor, label_tensor

def prep_data_classification_tcga(dat_tensor_train_orig, dat_tensor_valid_orig, label_df_train, label_df_valid, label_type, labels_mapping):
    
    dat_tensor_train, label_tensor_train = prep_data_label_remove_undefined(dat_tensor_train_orig, label_df_train, label_type, labels_mapping)
    dat_tensor_valid, label_tensor_valid = prep_data_label_remove_undefined(dat_tensor_valid_orig, label_df_valid, label_type, labels_mapping)
    
    return dat_tensor_train, dat_tensor_valid, label_tensor_train, label_tensor_valid
    
def standardise_and_merge(expr_df, cnv_df, snp_df):
    # Data Standardise + Merged 
    import pandas as pd
    
    expr_df_std = standardise(expr_df)

    frames=[expr_df_std, cnv_df, snp_df]
    data_df_orig = pd.concat(frames)
    data_df  = data_df_orig.transpose().copy()

    return data_df

def get_data_tensor(expr_df, cnv_df, snp_df):
    import torch

    data_df = standardise_and_merge(expr_df, cnv_df, snp_df)
    data_array = data_df.values
    data_tensor = torch.from_numpy(data_array).float()
    
    return data_tensor

def merge_data_3dfs(expr_df, cnv_df, snp_df):
    data_df = standardise_and_merge(expr_df, cnv_df, snp_df)
    return data_df

def merge_data_2dfs(expr_df, cnv_df):
    import pandas as pd
    
    expr_df_std = standardise(expr_df)
    frames=[expr_df_std, cnv_df]
    data_df_orig = pd.concat(frames)
    data_df  = data_df_orig.transpose().copy()
    
    return data_df

def convert_dat_df_to_tensor(data_df):
    import torch
    
    data_array = data_df.values
    data_tensor = torch.from_numpy(data_array).float()
    return data_tensor

def convert_label_df_to_tensor(label_df):
    import torch
    import numpy as np
    
    label_array  = label_df.values.astype(np.int32)
    label_tensor = torch.from_numpy(label_array).long().view(-1)
    return label_tensor

def convert_list_labels_to_tensor(list_label_df):
    return [convert_label_df_to_tensor(label_df) for label_df in list_label_df] 

def get_label_mapping(column):
    pam50_subtype_labels = {"Basal":0, "Her2":1, "LumA":2, "LumB":3, "Normal":4}
    tils_prediction = {1:1, 0:0}
    
    if column == "Pam50Subtype":
        subtype_labels = pam50_subtype_labels
    elif column == "tils_HandE_slides":
        subtype_labels = tils_prediction
    else:    
        subtype_labels = {"Positive":0, "Negative":1}
        
    return column, subtype_labels

def preprocess_two_datasets(tcga_df, metabric_df):
    
    tcga_df = tcga_df.loc[~tcga_df.index.duplicated(keep='first')]
    metabric_df = metabric_df.loc[~metabric_df.index.duplicated(keep='first')]
    
    tcga_df_rownames = tcga_df.index.tolist()
    metabric_df_rownames = metabric_df.index.tolist()
    genenames = sorted(list(set(tcga_df_rownames) & set(metabric_df_rownames)))
    
    tcga_df_final = tcga_df.loc[tcga_df.index.isin(genenames)].reindex(genenames)
    metabric_df_final = metabric_df.loc[metabric_df.index.isin(genenames)].reindex(genenames)
    
    return tcga_df_final, metabric_df_final

def get_label_mapping_file(column, map_file):
    import sys
    open_map_file = open(map_file)

    subtype_labels = {}

    for eachrow in open_map_file:
        eachrow = eachrow.strip().split("\t")
        file_column_name = eachrow[0]
        file_column_key  = eachrow[1]
        try:
            file_column_value= int(eachrow[2])
        except:
            print ("map_file third column has to be an integer")
            sys.exit(1)

        if file_column_name == column:
            subtype_labels.update({file_column_key:file_column_value})


    return column, subtype_labels

def rename_df_genenames(expr_df, cnv_df, snp_df):
    print_mult_data_shape(expr_df, cnv_df, snp_df, "Original")
    
    expr_df_final = expr_df.copy()
    cnv_df_final  = cnv_df.copy()
    snp_df_final  = snp_df.copy()
    
    rename_feature(expr_df_final, "_EXPR")
    rename_feature(cnv_df_final, "_CNV")
    rename_feature(snp_df_final, "_SNP")
    
    print_mult_data_shape(expr_df_final, cnv_df_final, snp_df_final, "PreProcess")
    
    return expr_df_final, cnv_df_final, snp_df_final  

def filter_clin_df_byColumn(clin_df, column):
    return clin_df[~clin_df[column].isna()]

def filter_clin_df_subtype(clin_df, subtype_list):
    return clin_df[clin_df["Pam50Subtype"].isin(subtype_list)]

def preprocess_dat_clin_df_byColumn(clin_df_orig, dat_df_orig, column):
    dat_df = dat_df_orig.transpose().copy()
    clin_df = clin_df_orig.copy()
    print_shape(clin_df, "Original", "ClinData")  
    clin_df.set_index("SampleName", inplace=True)
    
    clin_df = filter_clin_df_byColumn(clin_df, column)
    
    samplenames = clin_df.index.tolist()
    samplenames_dat = dat_df.columns.values
    samplenames_intersection = list(set(samplenames) & set(samplenames_dat))
    clin_df_final = clin_df.loc[clin_df.index.isin(samplenames_intersection)]
    dat_df_final  = dat_df[samplenames_intersection]
    clin_df_final = clin_df_final.reindex(dat_df_final.transpose().index)
    
    print_shape(clin_df_final, "PreProcess", "ClinData")
    print_shape(dat_df_final, "PreProcess", "MainData")
    
    return clin_df_final, dat_df_final.transpose()

def preprocess_dat_clin_df(clin_df_orig, dat_df_orig, subtype_list):
    dat_df = dat_df_orig.transpose().copy()
    clin_df = clin_df_orig.copy()
    print_shape(clin_df, "Original", "ClinData")  
    clin_df.set_index("SampleName", inplace=True)
    
    clin_df = filter_clin_df_subtype(clin_df, subtype_list)
    
    samplenames = clin_df.index.tolist()
    samplenames_dat = dat_df.columns.values
    samplenames_intersection = list(set(samplenames) & set(samplenames_dat))
    clin_df_final = clin_df.loc[clin_df.index.isin(samplenames_intersection)]
    dat_df_final  = dat_df[samplenames_intersection]
    clin_df_final = clin_df_final.reindex(dat_df_final.transpose().index)
    
    print_shape(clin_df_final, "PreProcess", "ClinData")
    print_shape(dat_df_final, "PreProcess", "MainData")
    
    return clin_df_final, dat_df_final.transpose()

def prep_data_label_remove_undefined2(dat_tensor_src, label_df_src, labels): #label_type, labels_mapping
    import pandas as pd
    import torch 
    import numpy as np
    
    dat_tensor_w = dat_tensor_src
    label_df_w = label_df_src.copy()
    
    list_labels = []
    list_labels_tensor = []
    
    print (labels)
    
    # Find all NAs / Undefined values that we need to clean
    # All samples ID collected to 'undefined_list'
    # Across multiple labels that we need to train (labels)
    undefined_list = []
    for label_id in labels:
        label_type = labels[label_id][0]
        
        # Get rid of Undefined
        if label_type == "Pam50Subtype":
            NAs = "Undefined"
        else:    
            NAs = "NA"
        
        undefined_list += label_df_w.index[label_df_w[label_type] == NAs].tolist()
        undefined_list += label_df_w.index[label_df_w[label_type].isna()].tolist()
        
        undefined_list = list(set(undefined_list))
        #print ("undefined_list", len(undefined_list), undefined_list)
        
    # Remove samples in 'undefined list' from data df
    # Making sure orders of indices are the same between data and labels 
    dat_df = pd.DataFrame(data = dat_tensor_w.detach().numpy(), index=label_df_w.index)    
    dat_filtered_df = dat_df.loc[~dat_df.index.isin(undefined_list)]
    #print ("dat_filtered_df", dat_filtered_df.shape, dat_filtered_df)
    
    # Remove samples in 'undefined list' from labels df too
    # Swap the mapping from characters to numeric (e.g. Positive -> 0; Negative->1)
    for label_id in labels:
        label_type = labels[label_id][0]
        labels_mapping = labels[label_id][1]
        label_filtered_df = label_df_w.loc[~label_df_w.index.isin(undefined_list)][label_type].replace(labels_mapping) 
        #print ("label_fitered_df", label_filtered_df.shape, label_filtered_df)
        list_labels.append(label_filtered_df)
    
    # convert to pytorch tensor
    dat_tensor = torch.from_numpy(dat_filtered_df.values).float() 
    print ("dat_tensor shape:", dat_tensor.shape)
    
    for label_filtered_df in list_labels:
        label_tensor = torch.from_numpy(label_filtered_df.values.astype(np.int32)).long().view(-1)
        list_labels_tensor.append(label_tensor)
        print("label_tensor shape:", label_tensor.shape)
        
    return dat_tensor, list_labels_tensor

def prep_data_classification2(dat_tensor_train_orig, dat_tensor_valid_orig, label_df_train, label_df_valid, labels):
    
    print (dat_tensor_train_orig.shape, label_df_train.shape)
    print (dat_tensor_valid_orig.shape, label_df_valid.shape)
    
    dat_tensor_train, list_label_tensor_train = prep_data_label_remove_undefined2(dat_tensor_train_orig, label_df_train, labels)
    dat_tensor_valid, list_label_tensor_valid = prep_data_label_remove_undefined2(dat_tensor_valid_orig, label_df_valid, labels)
    
    return dat_tensor_train, dat_tensor_valid, list_label_tensor_train, list_label_tensor_valid

def prep_data_classification_single(dat_tensor, label_df, alllabels):
    
    print (dat_tensor.shape, label_df.shape)
    
    dat_tensor, list_label_tensor = prep_data_label_remove_undefined2(dat_tensor, label_df, alllabels)
    
    return dat_tensor, list_label_tensor

def split_samples_balance(clin_df, column, split=0.7):
    import random

    all_samples = clin_df.index.tolist()
    sample_numbers = int(len(all_samples) * split)
    
    all_classes = list(set(clin_df[column].tolist()))
    min_samples = min(clin_df[column].value_counts())
    max_samples = max(clin_df[column].value_counts())
    sample_numbers_per_class = int(min_samples * split)
    
    train_samples = []
    for label_class in all_classes:
        class_sample = clin_df[clin_df[column] == label_class].index.tolist()
        train_class_sample = random.sample(class_sample, k=sample_numbers_per_class)
        train_samples += train_class_sample
        
    test_samples = [x for x in all_samples if x not in train_samples]
    
    return train_samples, test_samples

def split_samples_random(clin_df, column, split=0.7):
    import random
    
    all_samples = clin_df.index.tolist()
    sample_numbers = int(len(all_samples) * split)
    
    random.seed(12)
    train_samples = random.sample(all_samples, k=sample_numbers)
    test_samples = [x for x in all_samples if x not in train_samples]
    
    return train_samples, test_samples

def datasets_split2(dat_df, clin_df, column, split, method):
    if method == "random":
        split_method = split_samples_random
    elif method == "balance":
        split_method = split_samples_balance
    
    train_samples, test_samples = split_method(clin_df, column, split)
    
    dat_df = dat_df.copy().transpose()
    
    dat_df_train = dat_df[train_samples]
    dat_df_test = dat_df[test_samples]
    
    clin_df_train = clin_df.loc[train_samples]
    clin_df_test = clin_df.loc[test_samples]
    
    return dat_df_train.transpose(), dat_df_test.transpose(), clin_df_train, clin_df_test

def filter_df_for_NAs(dat_df_src, label_df_src, labels): #label_type, labels_mapping
    import pandas as pd
    import torch 
    import numpy as np
    
    print_shape(dat_df_src, "dat_df", "Prep Prior to Final Filtering")
    print_shape(label_df_src, "label_df", "Prep Prior to Final Filtering")
    
    dat_df = dat_df_src.copy()
    label_df_w = label_df_src.copy()
    
    list_labels = []
    list_labels_tensor = []
    
    print (labels)
    
    # Find all NAs / Undefined values that we need to clean
    # All samples ID collected to 'undefined_list'
    # Across multiple labels that we need to train (labels)
    undefined_list = []
    for label_id in labels:
        label_type = labels[label_id][0]
        
        # Get rid of Undefined
        if label_type == "Pam50Subtype":
            NAs = "Undefined"
        else:    
            NAs = "NA"
        
        undefined_list += label_df_w.index[label_df_w[label_type] == NAs].tolist()
        undefined_list += label_df_w.index[label_df_w[label_type].isna()].tolist()
        
        undefined_list = list(set(undefined_list))
        #print ("undefined_list", len(undefined_list), undefined_list)
        
    # Remove samples in 'undefined list' from data df
    # Making sure orders of indices are the same between data and labels 
    dat_df = dat_df.reindex(label_df_w.index)    
    dat_filtered_df = dat_df.loc[~dat_df.index.isin(undefined_list)]
    #print ("dat_filtered_df", dat_filtered_df.shape, dat_filtered_df)
    
    # Remove samples in 'undefined list' from labels df too
    # Swap the mapping from characters to numeric (e.g. Positive -> 0; Negative->1)
    for label_id in labels:
        label_type = labels[label_id][0]
        labels_mapping = labels[label_id][1]
        label_filtered_df = label_df_w.loc[~label_df_w.index.isin(undefined_list)][label_type].replace(labels_mapping) 
        #print ("label_fitered_df", label_filtered_df.shape, label_filtered_df)
        list_labels.append(label_filtered_df)
    
    print_shape(dat_filtered_df, "dat_df", "Post Filtering")
    for each_label_df in list_labels:
        print_shape(each_label_df, "label_df", "Post Filtering")
    
    return dat_filtered_df, list_labels