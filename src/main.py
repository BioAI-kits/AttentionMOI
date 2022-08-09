import os
from .utils import check_files
from .features import selection
from .train import train
from .explain import explain

def run(args):
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents  
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # features selection
    omics, labels = selection(args)

    # training model
    train(omics, labels, args)

    # # explain model
    # explain(args)









