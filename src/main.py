import os
from .utils import check_files, init_log
from .fsd import feature_selection_distribution
from .train import train, ml_models, train_net
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
    elif args.model == "Net":
        model, dataset_test = train_net(args, data, feature_group, labels)
        # # explain model
        explain(args, model, dataset_test, feature_name, feature_group, labels)

    # multiple models
    elif args.model == "all":
        # DNN model
        model, dataset_test = train(args, data, labels)
        explain(args, model, dataset_test, feature_name, feature_group, labels, name="DNN")
        # Net model
        model, dataset_test = train_net(args, data, feature_group, labels)
        explain(args, model, dataset_test, feature_name, feature_group, labels, name="Net")
        # ml model
        for name in ["RF", "XGboost", "svm"]:
            ml_models(args, data, feature_name, feature_group, labels, model_name=name)











