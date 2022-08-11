import sys, os, argparse, warnings, time
import numpy as np
from src.main import run


warnings.filterwarnings('ignore')
np.random.seed(1234)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--omic_file', action='append', help='omics file.', required=True)
    parser.add_argument('-n','--omic_name', action='append', help='omics name.', required=True)
    parser.add_argument('-l','--label_file', help='label file', required=True)
    parser.add_argument('-o','--outdir', help='output dir.', type=str, default='./output')
    parser.add_argument('-c','--clin_file', default=None, help='clinical features')
    parser.add_argument('-i','--iteration', help='iteration number.', type=int, default=10)
    parser.add_argument('-s','--seed', help='seed.', type=int, default=0)

    parser.add_argument('-b','--batch', help='Mini-batch number.', type=int, default=32)
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=300)
    parser.add_argument('-r','--lr', help='learning rate.', type=float, default=0.0001)
    parser.add_argument('-t','--test_size', help='Testing data proportion.', type=float, default=0.3)



    # parser.add_argument('-m','--minfeat', help='min features number.', type=int, default=50)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args() 
    run(args)
