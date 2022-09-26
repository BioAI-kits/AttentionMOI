import pandas as pd
from sklearn.feature_selection import SelectPercentile, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ANOVA
def anova(args, df_omic, label):
    """
    args: parameters from deepmoi.py.
    df_omic: omic data with DataFrame format.
    label: sample label.

    Return:
        selected omic data with DataFrame format.
    """
    select = SelectPercentile(percentile=args.percentile)
    select.fit(X=df_omic.values, y=label)
    features = df_omic.columns.values[select.get_support()]  # features selected by ANOVA.
    return df_omic[features]

# Recursive Feature Elimination (RFE)
def rfe(args, df_omic, label):
    """
    args: parameters from deepmoi.py.
    df_omic: omic data with DataFrame format.
    label: sample label.

    Return:
        selected omic data with DataFrame format.
    """
    keep_feat_num = df_omic.shape[1] // 10  # 默认保留10%的特征
    select = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=args.seed),
                 n_features_to_select=keep_feat_num, step=0.01)
    select.fit(X=df_omic.values, y=label)
    features = df_omic.columns.values[select.get_support()]  # features selected by REF.
    return df_omic[features]


# LASSO
def lasso(df_omic, label):
    """
    args: parameters from deepmoi.py.
    df_omic: omic data with DataFrame format.
    label: sample label.

    Return:
        selected omic data with DataFrame format.
    """
    cs = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    select = LogisticRegressionCV(Cs=cs, cv=5, penalty='l1', max_iter=1000, solver='liblinear')
    select.fit(X=df_omic.values, y=label)
    features = df_omic.columns[select.coef_[0] != 0]
    return df_omic[features]


# PCA
def pca(args, df_omic):
    """
    args: parameters from deepmoi.py.
    df_omic: omic data with DataFrame format.
    label: sample label.

    Return:
        selected omic data with DataFrame format.
    """
    num_pc = args.num_pc
    features = df_omic.columns
    x_stand = StandardScaler().fit(df_omic).transform(df_omic)
    pca_transformer = PCA().fit(x_stand)
    x_pc = pca_transformer.transform(x_stand)
    var_explained = pca_transformer.explained_variance_ratio_
    print("Variance explained by the first 2 PCs is {}, \nVariance explained by the first 30 PCs is {}.".format(sum(var_explained[0:2]), sum(var_explained[0:30])))
    print("Variance explained by {} PCs is {}.".format(num_pc, sum(var_explained[0:num_pc])))
    df_omic = pd.DataFrame(x_pc[:, 0:num_pc])
    df_omic.columns = ["PC_" + str(i + 1) for i in range(len(df_omic.columns))]
    f_contribution = pca_transformer.components_[0:num_pc, :].sum(axis=0)
    feature_imp = pd.DataFrame(data=f_contribution, index=features, columns=["feature_imp"])
    return df_omic, feature_imp
