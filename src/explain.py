from .utils import ig

def explain(args, model, dataset, feature_names, omic_group, labels):
    print('Perform model interpretation')
    for target in set(labels):
        ig(args, model, dataset, feature_names, omic_group, target=int(target))


