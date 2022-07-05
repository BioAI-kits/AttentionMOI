import sys, os, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import Set2Set, SumPooling
from data import read_omics, read_clin, build_graph, read_pathways
from util import evaluate, check_files
from module import DeepMOI
from train import train


warnings.filterwarnings('ignore')
np.random.seed(1234)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--omic_file', action='append', help='omics file.', required=True)
    parser.add_argument('-l','--label_file', help='label file', required=True)
    parser.add_argument('-a','--additional_features', default=None, help='Non-omic features')
    parser.add_argument('-p','--pathway', help='The pathway file that should be gmt format.', type=str, default='default')
    parser.add_argument('-n','--network', help='The network file that should be gmt format.', type=str, default='default')
    parser.add_argument('-b','--batch', help='Mini-batch number.', type=int, default=16)
    parser.add_argument('-e','--epoch', help='epoch number.', type=int, default=50)
    parser.add_argument('-r','--lr', help='learning rate.', type=float, default=0.001)
    args = parser.parse_args()
    
    print(args.omic_file)
    # # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # # Running main function
    train(omics_files=args.omic_file, 
          label_file=args.label_file, 
          add_file=args.additional_features,
          test_size=0.2,
          pathway_file=args.pathway,
          network_file=args.network,
          minibatch=args.batch, 
          epoch=args.epoch,
          lr=args.lr 
          )

    
    # print("[INFO] DeepMOI has finished running!")

