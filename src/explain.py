from .utils import ig, ig_net

def explain(args, model, dataset, feature_names, omic_group, labels, name=None):
    print('Perform model interpretation')
    if args.model == "DNN" or name == "DNN":
        for target in set(labels):
            ig(args, model, dataset, feature_names, omic_group, target=int(target))
    if args.model == "Net" or name == "Net":
        for target in set(labels):
            ig_net(args, model, dataset, feature_names, omic_group, target=int(target))


