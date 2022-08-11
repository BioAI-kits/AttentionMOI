import os
from .utils import check_files
from .selection import feature_selection
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
    dataset, feature_name, feature_group = feature_selection(args)

    # training model
    model, dataset_test = train(args, dataset)

    # # explain model
    explain(args, model, dataset_test, feature_name, feature_group)









