import sys, os, argparse, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import *
from util import *
from train import *
from module import DeepMOI


warnings.filterwarnings('ignore')
np.random.seed(1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--omic_file', action='append', help='omics file.', required=True)
    parser.add_argument('-l','--label_file', help='label file', required=True)
    parser.add_argument('-o','--outdir', help='output dir.', type=str, default='./output')
    parser.add_argument('-a','--additional_features', default=None, help='Non-omic features')
    parser.add_argument('-p','--pathway', help='The pathway file that should be gmt format.', type=str, default='default')
    parser.add_argument('-n','--network', help='The network file that should be gmt format.', type=str, default='default')
    parser.add_argument('-b','--batch', help='Mini-batch number.', type=int, default=16)
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=50)
    parser.add_argument('-r','--lr', help='learning rate.', type=float, default=0.001)
    parser.add_argument('-t','--test_size', help='Testing data proportion.', type=float, default=0.3)
    
    args = parser.parse_args()
    
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # Running main function
    train(omics_files=args.omic_file, 
          label_file=args.label_file, 
          add_file=args.additional_features,
          test_size=args.test_size,
          pathway_file=args.pathway,
          network_file=args.network,
          minibatch=args.batch, 
          epoch=args.epoch,
          lr=args.lr,
          outdir=args.outdir
          )

    
    # print("[INFO] DeepMOI has finished running!")

