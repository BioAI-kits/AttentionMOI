## A Denoised Multi-omics Integration Framework for Cancer Subtype Classification and Survival Prediction

---

### What we do?

- We integrated commonly used omics feature selection methods, such as ANOVA, RFE, LASSO, PCA.

- We also developed a new method for omics feature selection, FSD.

- we integrated common machine/deep learning algorithms used to build single/multi-omics model, such as DNN, Random Forest, XGboost, SVM, Mogonet$^1$, Moanna$^2$, which are two good multi-omics integration algorithms recently.

- We also developed a new deep learning algorithm for multi-omics integration.


---

<div align=center>
<img src="https://github.com/BioAI-kits/AttentionMOI/blob/master/img/Figure1.png" />
</div>

**Introduction of project**. The availability of high-throughput sequencing data create opportunities to comprehensively understand human diseases as well as challenges to train machine learning models using such high dimensions of data. Here, we propose a distribution based feature denosing algorithm, Feature Selection with Distribution (FSD), for multi-omics denosing to reduce dimensions of features and a multi-omics integration framework, Attention Multi-Omics Integration (AttentionMOI), which is inspired by the central dogma of biology. We demonstrated that FSD improved model performance either using single omics data or multi-omics data in 13 TCGA cancers for survival prediction and kidney cancer subtype identification. And our integration framework outperformed traditional artificial intellegnce models under high dimensions of features. Furthermore, FSD identisied features were related to cancer prognosis and could be considered as biomarkers. 

---

### Install

You can install programs and dependencies via pip. We recommend using conda to build a virtual environment with python version 3.9 or higher.

(optional) Create a virtual environment

```bash
conda create -n env_moi python=3.9

conda activate env_moi  # Activate the environment
```

Install

```bash
pip install AttentionMOI
```

### Parameters
 
After your installation is complete, your computer terminal will contain a `moi` command. This is the only interface to our program. You will use this command to build an omics model.

First, you can execute the following command line to get detailed help information.

```
moi -h
```

Then, we also introduce these parameters in the following documents: 


**1. Input**


f | omic_file

> REQUIRED: File path for omics files (should be matrix)

n | omic_name

> REQUIRED: Omic names for omics files, should be the same order as the omics file

l | label_file

> REQUIRED: File path for label file


**2. Output**

o | outdir

> OPTIONAL: Setting output file path, default=./output


**3. Feature selection**

method

> OPTIONAL: Method of feature selection, choosing from ANOVA, RFE, LASSO, PCA, default is no feature selection

percentile

> OPTIONAL: Percent of features to keep for ANOVA (integer between 1-100), only used when using ANOVA, default=30

num_pc

> OPTIONAL: Number of PCs to keep for PCA (integer), only used when using PCA, default=50

FSD

> OPTIONAL: Whether to use FSD to mitigate noise of omics. Default is not using FSD, and set --FSD to use FSD

i | iteration

> OPTIONAL: The number of FSD iterations (integer), default=10.

s | seed

> OPTIONAL: Random seed for FSD (integer), default=0

threshold

> OPTIONAL: FSD threshold to select features (float), default=0.8 (select features that are selected in 80 percent FSD iterations)


**4. Building Model**

m | model 

> OPTIONAL: Model names, choosing from DNN, Net (Net for AttentionMOI), RF, XGboost, svm, mogonet, moanna, default=DNN.

t | test_size

> OPTIONAL: Testing dataset proportion when split train test dataset (float), default=0.3 (30 percent data for testing)

b | batch

> OPTIONAL: Mini-batch number for model training (integer), default=32

e | epoch

> OPTIONAL: Epoch number for model training (integer), default=300

r | lr

> OPTIONAL: Learning rate for model training(float), default=0.0001.

w | weight_decay

> OPTIONAL: weight_decay parameter for model training (float), default=0.0001

---

### Example

Example (Data can be downloaded from https://github.com/BioAI-kits/AttentionMOI ): 
```
moi -f GBM_exp.csv.gz -f GBM_met.csv.gz -f GBM_logRatio.csv.gz -n rna -n met -n cnv -l GBM_label.csv --FSD -m all -o GBM_Result \n
```

---

### Ref.

1. MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification

2. Moanna: Multi-Omics Autoencoder-Based Neural Network Algorithm for Predicting Breast Cancer Subtypes 


---

All rights reserved.



