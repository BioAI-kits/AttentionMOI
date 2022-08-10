import os
from .utils import check_files
from .selection import feature_selection
from .train import train

def run(args):
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents  
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # features selection
    dataset = feature_selection(args)

    # training model
    train(args, dataset)

    # # explain model
    # explain(args)









