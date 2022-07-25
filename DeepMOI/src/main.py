import os
from .utils import check_files
from .train import train
from .explain import explain

def run(args):
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents  
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # training model
    train(args)

    # explain model
    explain(args)









