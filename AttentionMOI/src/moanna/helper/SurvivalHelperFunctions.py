def map_label(df, column, mapping):
    df[column] = df[column].replace(mapping)
    return df

def map_mult_labels(df, columns, labels):
    for column in columns:
        df = map_label(df, labels[column][0], labels[column][1])
    
    return df

#https://stackoverflow.com/questions/26886653/pandas-create-new-column-based-on-values-from-other-columns-apply-a-function-o
def surv_event(row, surv_year):
    if row['Event'] in ["DECEASED", "Recurred/Progressed", "PROGRESSION"]:
        if row['Time'] <= surv_year:
            return 1
        else:
            return 0
    else:
        return 0
    
def do_survival(clin_df, type_surv, labels_mapping, surv_year=120, filename="survival.png"): #type=[OS, DFS, RFS]
    import pandas as pd
    import matplotlib.pyplot as plt

    
    #header = ["ERStatus", "HER2Status", "Pam50Subtype", "Age", "Time", "Event"]
    header = ["Pam50Subtype", "Time", "Event"]
    df_surv = pd.DataFrame(index=clin_df.index, columns=header)
    #df_surv.ERStatus = clin_df.loc[df_surv.index, ['ERStatus']]
    #df_surv.HER2Status = clin_df.loc[df_surv.index, ['HER2Status']]
    df_surv.Pam50Subtype = clin_df.loc[df_surv.index, ['Pam50Subtype']]
    #df_surv.Age = clin_df.loc[df_surv.index, ['Age']]
    if type_surv == "OS":
        df_surv.Time = clin_df.loc[df_surv.index, ['OS_MONTHS']]
        df_surv.Event = clin_df.loc[df_surv.index, ['OS_STATUS']]
    elif type_surv == "DFS":
        df_surv.Time = clin_df.loc[df_surv.index, ['DFS_MONTHS']]
        df_surv.Event = clin_df.loc[df_surv.index, ['DFS_STATUS']]
    elif type_surv == "RFS":
        df_surv.Time = clin_df.loc[df_surv.index, ['RFS_MONTHS']]
        df_surv.Event = clin_df.loc[df_surv.index, ['RFS_STATUS']]    
    
    df_surv["SurvEvent"]=df_surv.apply(lambda row: surv_event(row, surv_year), axis=1)
    df_surv.loc[df_surv['Time']>surv_year, 'SurvTime'] = surv_year
    df_surv.loc[df_surv['Time']<=surv_year, 'SurvTime'] = df_surv['Time']

    df_surv_final = df_surv.drop(['Time', 'Event'], axis=1)
    print (df_surv_final.shape)
    print (sum(df_surv_final.SurvTime.isna()))
    df_surv_final = df_surv_final[~df_surv_final.SurvTime.isna()]
    print (sum(df_surv_final.SurvTime.isna()))
    print (df_surv_final.shape)
    
    from lifelines import CoxPHFitter
    cph = CoxPHFitter()
    cph.fit(df_surv_final, duration_col='SurvTime', event_col='SurvEvent')
    cph.print_summary() 
    
    #cph.plot()
    #cph.plot_covariate_groups('ERStatus', [3,2,1,0])
    
    from lifelines import KaplanMeierFitter
    kmf = KaplanMeierFitter()
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    for name, grouped_df in df_surv_final.groupby('Pam50Subtype'):
        print (name)
        kmf.fit(grouped_df["SurvTime"], grouped_df["SurvEvent"], label=name)
        kmf.plot(ax=ax, ci_show=False, linewidth=4, color=['firebrick', 'hotpink', 'darkblue', 'aqua'][name])
        
    ax.legend(labels_mapping)
    plt.savefig(filename)
    return cph, kmf       