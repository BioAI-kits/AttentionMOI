import argparse, warnings
import numpy as np
from src.main import run

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
    parser.add_argument('-i', '--iteration', help='The number of FSD iterations.', type=int, default=10)
    parser.add_argument('-s', '--seed', help='Default random number seed to use for FSD.', type=int, default=0)
    parser.add_argument('--threshold',
                        help='Threshold to use for FSD screening features,i.e. the proportion of iterations through FSD.',
                        type=float, default=0.8)

    # feature selection
    parser.add_argument('--method', help='Method of feature selection, choosing from ANOVA, RFE, LASSO, PCA.', type=str, default=None)
    parser.add_argument('--percentile', help='percentile for ANOVA.', type=float, default=30)
    parser.add_argument('--num_pc', help='number of PCs for PCA.', type=int, default=50)

    # whether using FSD
    parser.add_argument('--FSD', action="store_true", help='whether to use FSD to mitigate noise of omics. If set --FSD then using FSD, else not using FSD.')

    # building model
    parser.add_argument('-t', '--test_size', help='Testing dataset proportion.', type=float, default=0.3)
    parser.add_argument('-b', '--batch', help='Mini-batch number for model training.', type=int, default=32)
    parser.add_argument('-e', '--epoch', help='Epoch number for model training.', type=int, default=300)
    parser.add_argument('-r', '--lr', help='Learning rate for model training.', type=float, default=0.0001)
    parser.add_argument('-w', '--weight_decay', help='weight_decay parameter for model training.', type=float,
                        default=0.0001)

    # different models
    parser.add_argument('-m', '--model', help='Model names, choosing from RF, XGboost, svm, DNN, Net，all', type=str, default="DNN")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    run(args)

