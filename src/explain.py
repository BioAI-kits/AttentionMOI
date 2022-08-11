import torch
from captum.attr import IntegratedGradients
import pandas as pd
import numpy as np
import os

def explain(args, model, dataset, feature_names, omic_group):
    # prepare input data
    input_tensor_list = [d[0].unsqueeze(0) for d in dataset]
    input_tensor = torch.cat(input_tensor_list, 0)
    input_tensor.requires_grad_()

    # instantiation
    ig = IntegratedGradients(model)
    
    # calculating feature importance using IG
    attr, _ = ig.attribute(input_tensor, return_convergence_delta=True)
    attr = attr.detach().numpy()
    feat_importance = np.mean(attr, axis=0)

    # result
    df_imp = pd.DataFrame({'Feature': feature_names, 
                           'Omic': omic_group,
                           'Importance_value': feat_importance,
                           'Importance_value_abs': abs(feat_importance)
                           })
    df_imp = df_imp.sort_values('Importance_value_abs', ascending=False)

    # output 
    df_imp.to_csv(os.path.join(args.outdir, 'feature_importance.csv'), index=False)
    
    return df_imp


