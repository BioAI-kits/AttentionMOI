import os
from .utils import check_files, init_log
from .fsd import feature_selection_distribution
from .train import train, ml_models
from .explain import explain


def run(args):
    # check files exists
    check_files(args.omic_file)
    check_files(args.label_file)

    # make documents
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # init log.txt
    init_log(args=args)

    # features selection - with FSD or not
    data, feature_name, feature_group, labels = feature_selection_distribution(args)

    # training model - using different models
    if args.model == "DNN":
        model, dataset_test = train(args, data, labels)
        # # explain model
        explain(args, model, dataset_test, feature_name, feature_group, labels)
    elif args.model in ["RF", "XGboost", "svm"]:
        name = args.model
        ml_models(args, data, feature_name, feature_group, labels, model_name=name)

    # multiple models
    elif args.model == "all":
        # DNN model
        model, dataset_test = train(args, data, labels)
        # # explain model
        explain(args, model, dataset_test, feature_name, feature_group, labels)
        # ml model
        for name in ["RF", "XGboost", "svm"]:
            ml_models(args, data, feature_name, feature_group, labels, model_name=name)











