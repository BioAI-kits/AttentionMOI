## A Denoised Multi-omics Integration Framework for Cancer Subtype Classification and Survival Prediction

Jiali Panga, Bilin Lianga, Ruifeng Dingb, Qiujuan Yana, Ruiyao Chena, Jie Xu* 

### Introduction of project

The availability of high-throughput sequencing data create opportunities to comprehensively understand human diseases as well as challenges to train machine learning models using such high dimensions of data. Here, we propose a distribution based feature denosing algorithm, Feature Selection with Distribution (FSD), for multi-omics denosing to reduce dimensions of features and a multi-omics integration framework, Attention Multi-Omics Integration (AttentionMOI), which is inspired by the central dogma of biology. We demonstrated that FSD improved model performance either using single omics data or multi-omics data in 13 TCGA cancers for survival prediction and kidney cancer subtype identification. And our integration framework outperformed traditional artificial intellegnce models under high dimensions of features. Furthermore, FSD identisied features were related to cancer prognosis and could be considered as biomarkers. 

<div align=center>
<img src="https://github.com/BioAI-kits/AttentionMOI/blob/master/img/Figure1.png" />
</div>

---

### Dependence

To use the project codes, some dependences should be installed firstly:

```
pip install -r requirements.txt
```

### To perform FSD and build models

The program performs feature selection and model building through deepmoi.py files.

Examples:

```
python moi.py -f ./dataset/GBM/GBM_exp.csv.gz \
              -f ./dataset/GBM/GBM_met.csv.gz \
              -f ./dataset/GBM/GBM_logRatio.csv.gz \
              -l ./dataset/GBM/GBM_label.csv \
              -n rna \
              -n met \
              -n cnv \
              -b 16 \
              --FSD \
              -m all \
              --threshold 0.2 \
              -o ./GBM_output
```

More detailed parameter descriptions can be obtained in the following ways:

```
python moi.py -h
```


All rights reserved.



