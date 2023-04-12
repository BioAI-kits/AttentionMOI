def plot_scatter(tsne_comp, samplenames, clin_df, colname, filename): #colname=Pam50Subtype
    import pandas as pd
    import matplotlib.pyplot as plt
    
    import matplotlib.pylab as pylab
    params = {'legend.fontsize': 'x-large',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
         'axes.labelweight': 'bold',
             'font.weight' : 'bold',
              'axes.facecolor': 'white',}
    pylab.rcParams.update(params)    
    
    plot_df = pd.DataFrame(data=tsne_comp, 
                          columns = ['comp 1', 'comp 2'], index=samplenames)
    plot_df = pd.concat([plot_df, clin_df[colname]], axis=1)
    print(plot_df.shape)

    targets = [x for x in list(plot_df[colname].unique()) if str(x) != 'nan']
    targets.sort()
    print(targets)
    colors = ['firebrick', 'hotpink', 'darkblue', 'aqua', 'm', 'y', 'b'][0:len(targets)]

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(1,1,1)
    #ax.set_xlabel('Component 1', fontsize = 15)
    #ax.set_ylabel('Component 2', fontsize = 15)
    
    if colname == "Pam50Subtype":
        ax.set_title("PAM50 Subtypes", fontsize = 20)
    else:
        ax.set_title(colname, fontsize = 20)

    for target, color in zip(targets,colors):
        indicesToKeep = plot_df[colname] == target
        ax.scatter(plot_df.loc[indicesToKeep, 'comp 1']
                , plot_df.loc[indicesToKeep, 'comp 2']
                , c = color
                , s = 40
                , alpha = 1)
    
    ax.legend(targets, fontsize=20)    
    ax.grid()
    plt.savefig(filename)
    
def plot_scatter_kmeans(tsne_comp, samplenames, kmeans_group):
    import pandas as pd
    import matplotlib.pyplot as plt
    
    plot_df = pd.DataFrame(data=tsne_comp, 
                          columns = ['comp 1', 'comp 2'], index=samplenames)
    plot_df['clusters'] = kmeans_group

    colname='clusters'
    targets = [x for x in list(plot_df[colname].unique()) if str(x) != 'nan']
    print(targets)
    colors = ['r', 'hotpink', 'darkblue', 'aqua', 'm', 'y', 'b'][0:len(targets)]

    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(1,1,1)
    #ax.set_xlabel('PC1', fontsize = 15)
    #ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title(colname, fontsize = 20)

    for target, color in zip(targets,colors):
        indicesToKeep = plot_df[colname] == target
        ax.scatter(plot_df.loc[indicesToKeep, 'comp 1']
                , plot_df.loc[indicesToKeep, 'comp 2']
                , c = color
                , s = 10
                , alpha = 0.8)
    
    ax.legend(targets, fontsize=20)    
    ax.grid()
    
def do_TSNE(dat_df, clin_df, column, plot_name):
    from sklearn.manifold import TSNE
    tsne_comp = TSNE().fit_transform(dat_df)
    
    samplenames = clin_df.index.tolist()
    plot_scatter(tsne_comp, samplenames, clin_df, column, plot_name)
    
    return tsne_comp, samplenames

def clustering_metrics(y_true, y_pred):
    from sklearn import metrics
    
    metrics_list = {'ARI': metrics.adjusted_rand_score, 
                    'NMI': metrics.normalized_mutual_info_score, 
                    'FM': metrics.fowlkes_mallows_score
                   }
    
    metrics_dict = {}
    for m, comp_fn in metrics_list.items():
        #print (m, comp_fn(y_true, y_pred))
        metrics_dict[m] = comp_fn(y_true, y_pred)
        
    return metrics_dict 