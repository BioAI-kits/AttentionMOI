import os, sys
import numpy as np
import pandas as pd 
import torch
from pathlib import Path

from model.Moanna import Moanna
from helper.DataCleanUpFunctions import convert_dat_df_to_tensor, convert_list_labels_to_tensor

def moanna_predict(input, out_summary, out_details, model_state, model_json):

    base_path = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    moanna = load_model(model_state, model_json)
    moanna.eval()

    # Read Input
    dat_df = pd.read_csv(input, sep="\t", index_col=0)

    # Process Input
    dat_tensor = convert_dat_df_to_tensor(dat_df)
    dat_tensor = dat_tensor.to(device)

    # Run Moanna
    encoded, decoded, predictions = moanna(dat_tensor)

    # Get Output
    ## TODO: Parameterised
    type_of_predictions={0:"ERStatus", 1:"HER2Status", 2: "Pam50Subtype"}

    pred_class_dict = {}
    pred_percent_list = []

    for i, label in type_of_predictions.items():
        # Get prediction labels and its string mapping
        mapping_file = os.path.join(base_path, "mapping_file", label+".txt")
        labels_map   = get_label_mapping(label, mapping_file)
        all_labels   = list(labels_map.values())

        # Get prediction 
        pred_classes = get_pred_classes(predictions[i])
        pred_percent_list.append(pd.DataFrame(get_pred_percentage(predictions[i]), index=dat_df.index.to_list(), columns=all_labels))

        # Map prediction label to string
        pred_class_dict["Pred_" + label] = [labels_map [x] for x in pred_classes]

    pred_class_df = pd.DataFrame(pred_class_dict, index=dat_df.index.to_list())
    pred_percent_df = pd.concat(pred_percent_list, axis=1, sort=False)

    # Write output
    pred_class_df.to_csv(out_summary, sep="\t")
    pred_percent_df.to_csv(out_details, sep="\t")

    return 0

def moanna_train(input=None, label=None, output='test'):

    # Basic setup
    base_path = Path(__file__).parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read Inputs & Labels 
    # dat_df = pd.read_csv(input, sep="\t", index_col=0)
    df1 = pd.read_csv(r'C:\Users\liangbilin\Desktop\AttentionMOI\dataset\BLCA_select_data\BLCA_cnv.csv', index_col=0)
    df2 = pd.read_csv(r'C:\Users\liangbilin\Desktop\AttentionMOI\dataset\BLCA_select_data\BLCA_met.csv', index_col=0)
    df3 = pd.read_csv(r'C:\Users\liangbilin\Desktop\AttentionMOI\dataset\BLCA_select_data\BLCA_rna.csv', index_col=0)   
    patients = df1.index.values.tolist() + df2.index.values.tolist() + df3.index.values.tolist()
    patients_count = pd.DataFrame(pd.value_counts(patients))
    patients_extracted = patients_count[patients_count.iloc[:, 0]==3].index.values
    dat_df = pd.concat( [df1.loc[patients_extracted,:], df2.loc[patients_extracted,:], df3.loc[patients_extracted,:]], axis=1)

    # labels_df = pd.read_csv(label, sep="\t", index_col=0)
    # list_label_df = [labels_df[col] for col in labels_df.columns.tolist()]
    list_label_df = pd.read_csv(r'C:\Users\liangbilin\Desktop\AttentionMOI\dataset\BLCA_select_data\BLCA_label.csv').label.values.tolist()

    # Pre-defined hyperparameters
    ## TODO: Parameterised 
    params = {}
    params["input_size"] = dat_df.shape[1] # Number of features
    params["n_layers"] = 1
    params["encoded_size"] = 64 # Number of Autoencoders neurons
    params["hidden_size"] = params["encoded_size"] * (2 ** (params["n_layers"] + 1))
    params["batch_size"] = dat_df.shape[0] # Number of samples
    params["drop_prob"] = 0.5
    params["fnn_hidden_size"] = 40
    params["fnn_number_layers"] = 2
    params["num_classes"] = [len(np.unique(classes)) for classes in list_label_df]
    params["fnn_num_epoch"] = 100
    params["fnn_learning_rate"] = 0.05

    # Setup model
    moanna = Moanna(
        params["input_size"], 
        params["hidden_size"], 
        params["encoded_size"], 
        params["n_layers"], 
        params["drop_prob"], 
        params["fnn_hidden_size"], 
        params["num_classes"], 
        params["fnn_number_layers"], 
        0.1,
    )
    moanna = moanna.to(device)

    # Training optimiser
    moanna_optimizer = torch.optim.SGD(moanna.parameters(), lr=params["fnn_learning_rate"] , momentum=0.9)

    # Loss functions
    criterion1 = torch.nn.MSELoss(reduce=True)
    classifiers_criterion = []
    for classes in params["num_classes"]:
        classifiers_criterion.append(torch.nn.CrossEntropyLoss())   

    # Prep inputs
    dat_tensor = convert_dat_df_to_tensor(dat_df)
    # list_label_tensor = convert_list_labels_to_tensor(list_label_df) 
    list_label_tensor = torch.Tensor(list_label_df).reshape(-1,1)

    # For GPU's training
    dat_tensor = dat_tensor.to(device)
    for i in range(len(list_label_tensor)):
        list_label_tensor[i] = list_label_tensor[i].to(device)   

    # Train model
    for epoch in range(params["fnn_num_epoch"]):
        moanna_optimizer.zero_grad()
        encoded, decoded, predictions = moanna(dat_tensor)
    
        losses = [criterion1(decoded, dat_tensor)]
        for i in range(len(predictions)):
            print(len(predictions), list_label_tensor[i])
            losses.append(classifiers_criterion[i](predictions[i], list_label_tensor[i]))
        loss  = sum(losses)
    
        print ("Epoch %s: loss=%s" %(epoch,loss))
    
        loss.backward()
        moanna_optimizer.step()

    # Save model
    save_model(moanna, output, params)
    
    return 0

def get_pred_percentage(pred):
    val = torch.softmax(pred, dim=1)
    val_decimal = val.cpu().detach().numpy()
    val_percent = pd.DataFrame(val_decimal).applymap(lambda x: '{:.2%}'.format(x)).values
    return val_percent
    
def get_pred_classes(pred):
    val = torch.softmax(pred, dim=1)
    _, _classes = torch.max(val, 1)  
    classes = _classes.cpu().numpy()
    return classes

def get_pred_max(pred):
    val = torch.softmax(pred, dim=1)
    val_decimal = val.cpu().detach().numpy()  
    val_percent = pd.DataFrame(val_decimal.max(axis=1)).applymap(lambda x: '{:.2%}'.format(x)).values
    return val_percent

def get_label_mapping(column, mapping_file):
    '''
    PyTorch label is numeric and this function maps it back to a string label
    '''
    open_map_file = open(mapping_file)

    subtype_labels = {}

    for eachrow in open_map_file:
        eachrow = eachrow.strip().split("\t")
        file_column_name = eachrow[0]
        file_column_key  = eachrow[1]
        try:
            file_column_value= int(eachrow[2])
            file_column_label= eachrow[3]
        except:
            print ("map_file third column has to be an integer")
            sys.exit(1)

        if file_column_name == column:
            subtype_labels.update({file_column_value:file_column_label})

    return subtype_labels

def save_model(model, output, params):
    import json

    torch.save(model.state_dict(), output+".pt")

    with open(output+".json", 'w') as file:
        json.dump(params, file)

def load_model(model_state, model_json):
    import json

    with open(model_json, 'r') as file:
        params = json.load(file)      

    model = Moanna(
        params["input_size"], 
        params["hidden_size"], 
        params["encoded_size"], 
        params["n_layers"], 
        params["drop_prob"], 
        params["fnn_hidden_size"], 
        params["num_classes"], 
        params["fnn_number_layers"], 
        0.1,
    )

    model.load_state_dict(torch.load(model_state))

    return model



if __name__ == '__main__':
    moanna_train()
