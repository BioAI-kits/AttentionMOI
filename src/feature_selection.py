from sklearn.feature_selection import SelectPercentile, RFE
from sklearn.ensemble import RandomForestClassifier

# ANOVA
def anova(args, df_omic, label):
    """
    args: paramters from deepmoi.py.
    df_omic: omic data with DataFrame format.
    label: sample label.

    Return:
        selected omic data with DataFrame format.
    """
    select = SelectPercentile(percentile=args.percentile)
    select.fit(X=df_omic.values, y=label)
    features = df_omic.columns.values[select.get_support()]  # features selected by ANOVA.
    return df_omic[features]


def rfe(args, df_omic, label):
    """
    args: paramters from deepmoi.py.
    df_omic: omic data with DataFrame format.
    label: sample label.

    Return:
        selected omic data with DataFrame format.
    """
    keep_feat_num = df_omic.shape[1] // 2  # 默认保留10%的特征
    select = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=args.seed),
                 n_features_to_select=keep_feat_num
                )
    select.fit(X=df_omic.values, y=label)
    features = df_omic.columns.values[select.get_support()]  # features selected by REF.
    return df_omic[features]













