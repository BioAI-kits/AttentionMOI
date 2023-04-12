import argparse, warnings, sys
import numpy as np
from .src.main import run

warnings.filterwarnings('ignore')
np.random.seed(1234)


def get_args():
    parser = argparse.ArgumentParser(
                                     prog='AttentionMOI',
                                     usage="The program is used to build machine/deep learning model with single/multi omics dataset.",
                                     description="", 
                                     epilog="Example (Data can be downloaded from https://github.com/BioAI-kits/AttentionMOI ): \nmoi -f GBM_exp.csv.gz -f GBM_met.csv.gz -f GBM_logRatio.csv.gz -n rna -n met -n cnv -l GBM_label.csv --FSD -m all -o GBM_Result \n ",
                                     formatter_class=argparse.RawTextHelpFormatter
                                     )

    # config
    parser.add_argument('-f', '--omic_file', action='append', help='REQUIRED: File path for omics files (should be matrix)', required=True)
    parser.add_argument('-n', '--omic_name', action='append',
                        help='REQUIRED: Omic names for omics files, should be the same order as the omics file', required=True)
    parser.add_argument('-l', '--label_file', help='REQUIRED: File path for label file', required=True)
    parser.add_argument('-o', '--outdir', help='OPTIONAL: Setting output file path, default=./output', type=str, default='./output')

    # feature selection with distribution
    parser.add_argument('-i', '--iteration', help='OPTIONAL: The number of FSD iterations (integer), default=10.', type=int, default=10)
    parser.add_argument('-s', '--seed', help='OPTIONAL: Random seed for FSD (integer), default=0', type=int, default=0)
    parser.add_argument('--threshold',
                        help='OPTIONAL: FSD threshold to select features (float), default=0.8 (select features that are selected in 80 percent FSD iterations)',
                        type=float, default=0.8)

    # feature selection
    parser.add_argument('--method', help='OPTIONAL: Method of feature selection, choosing from ANOVA, RFE, LASSO, PCA, default is no feature selection', type=str, default=None)
    parser.add_argument('--percentile', help='OPTIONAL: Percent of features to keep for ANOVA (integer between 1-100), only used when using ANOVA, default=30', type=int, default=30)
    parser.add_argument('--num_pc', help='OPTIONAL: Number of PCs to keep for PCA (integer), only used when using PCA, default=50', type=int, default=50)

    # whether using FSD
    parser.add_argument('--FSD', action="store_true", help='OPTIONAL: Whether to use FSD to mitigate noise of omics. Default is not using FSD, and set --FSD to use FSD')

    # building model
    parser.add_argument('-t', '--test_size', help='OPTIONAL: Testing dataset proportion when split train test dataset (float), default=0.3 (30 percent data for testing)', type=float, default=0.3)
    parser.add_argument('-b', '--batch', help='OPTIONAL: Mini-batch number for model training (integer), default=32', type=int, default=32)
    parser.add_argument('-e', '--epoch', help='OPTIONAL: Epoch number for model training (integer), default=300', type=int, default=300)
    parser.add_argument('-r', '--lr', help='OPTIONAL: Learning rate for model training(float), default=0.0001.', type=float, default=0.0001)
    parser.add_argument('-w', '--weight_decay', help='OPTIONAL: weight_decay parameter for model training (float), default=0.0001', type=float,
                        default=0.0001)

    # different models
    parser.add_argument('-m', '--model', help='OPTIONAL: Model names, choosing from DNN, Net (Net for AttentionMOI), RF, XGboost, svm, mogonet, moanna, default=DNN.', type=str, default="DNN")

    args = parser.parse_args()
    return args



def main():
    args = get_args()
    
    if len(set((args.omic_name))) < 2 and args.model in ['Net', 'all']:
        print('Single omic data cannot be used to construct the AttentionMOI model.')
        sys.exit(1)
        
    run(args)


if __name__ == "__main__":
    main()

