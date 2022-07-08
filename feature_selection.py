import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



############## feature selection ##############

number = 1000  # keep features for each omic
files = ['./data/LGG/miRNA_gene_mean.csv.gz',
         './data/LGG/met.csv.gz',
         './data/LGG/rna.csv.gz'
        ]
gene_names = [] # keep genes names
for f in files:
    df = pd.read_csv(f, compression='gzip')
    df = df.set_index('gene')
    df = df.T
    df = df.sort_index()
    df = df.fillna(0)
    
    df_label = pd.read_csv('./data/LGG/label.csv')
    df_label = df_label[df_label.patient_id.isin(df.index.values)]
    df_label = df_label.sort_values('patient_id')
    
    Y = df_label.label.values
    X = df.values
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)
    select = RFE(estimator=DecisionTreeClassifier(random_state=42, criterion='entropy'),
                n_features_to_select=number,
                 step=0.1
                )
    select.fit(X_train, Y_train)
    
    gene_names += list(df.columns.values[select.get_support()])

choose_genes = set(gene_names)


############## feature selection ##############

df_p = pd.read_csv('./Pathway/Rectome.pathway.csv')
keep_pathways = []
prop = []
for pathway, group in df_p.groupby('pathway'):
    genes = set(group.src.to_list() + group.dest.to_list())
    propotion = len(genes & choose_genes) / len(genes)
    prop.append(propotion)
    if propotion > 0.2:  # only keep those pathways whose propotion more than 0.2 
        keep_pathways.append(pathway)
df_p[df_p.pathway.isin(keep_pathways)].to_csv('./Pathway/Rectome.pathway.tmp.csv', index=False)


