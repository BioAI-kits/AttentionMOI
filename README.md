
# coding tree

```
DeepMOI /
    |-- dataset /                       # 存储数据集,每个case一个目录
        |-- LGG /
            |-- rna.csv.gz
            |-- cnv.csv.gz
            |-- met.csv.gz
            |-- clin.csv.gz
            |-- label.csv.gz
            |-- readme.md
        |-- GBM /
            |-- rna.csv.gz
            |-- cnv.csv.gz
            |-- met.csv.gz
            |-- clin.csv.gz
            |-- label.csv.gz
            |-- readme.md
    |-- model /                         # 存储模型,每个case一个目录
        |-- LGG /
            |-- model.pt
            |-- readme.md
        |-- GBM /
            |-- model.pt
            |-- readme.md
    |-- src /                           # 存储依赖脚本
        |-- preprocess.py                   # 读取数据
        |-- selection.py                     # 特征筛选
        |-- module.py                       # 构建模型
        |-- explain.py                      # 模型解释
        |-- train.py                        # 训练脚本
        |-- main.py                         # 执行脚本
        |-- utils.py                        # 其他脚本
        |-- plot.py                         # 绘图脚本
    |-- deepmoi.py                      # 主程序
    |-- README.md                       # 说明文档
    |-- requirement.txt                 # 依赖库
```

# Install

## create conda enviroment

```sh
conda create DeepMOI
conda activate DeepMOI
```

## install requirement packages

```py
pip install requirements.txt
```

## download DeepMOI

```
git clone git@github.com:BioAI-kits/DeepMOI.git
```

## perform demo

```py
python deepmoi.py -f ./dataset/Test/rna.csv.gz -f ./dataset/Test/met.csv.gz  -l ./dataset/Test/label.csv -n rna -n met -s 42 -b 16
```

```py
python deepmoi.py -f ./dataset/GBM/cnv.csv.gz -f ./dataset/GBM/rna.csv.gz -f ./dataset/GBM/met.csv.gz  -l ./dataset/GBM/labels.csv -c ./dataset/GBM/clin.csv -n cnv -n rna -n met
```

# parameters

- -f 

> omic file

- -l 

> label file

- c 

> clinical file

-n 

> omic name (need to correspond to -f)




