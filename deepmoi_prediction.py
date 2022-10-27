import argparse, warnings
import numpy as np
from src.prediction import predict

warnings.filterwarnings('ignore')
np.random.seed(1234)


def get_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('-f', '--omic_file', action='append', help='Input omics file.', required=True)
    parser.add_argument('-n', '--omic_name', action='append',
                        help='Input the omics name in the same order as the omics file.', required=True)
    parser.add_argument('-c', '--clin_file', default=None, help='Input clinical features file (optional)')
    parser.add_argument('-l', '--label_file', help='Input label file', required=True)
    parser.add_argument('-o', '--outdir', help='Setting output directory.', type=str, default='./output')

    # feature selection with distribution
    parser.add_argument('-s', '--seed', help='Default random number seed to use for FSD.', type=int, default=0)

    # feature selection
    parser.add_argument('--method', help='Method of feature selection, choosing from ANOVA, RFE, LASSO, PCA.', type=str, default=None)

    # whether using FSD
    parser.add_argument('--FSD', action="store_true", help='whether to use FSD to mitigate noise of omics. If set --FSD then using FSD, else not using FSD.')

    # different models
    parser.add_argument('-m', '--model', help='Model names, choosing from RF, XGboost, svm, DNN, Net，all', type=str, default="DNN")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    predict(args)

