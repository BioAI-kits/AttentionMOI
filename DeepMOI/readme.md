
# coding tree

```
DeepMOI /
    |-- dataset /                       # 存储数据集,每个case一个目录
        |-- LGG /
            |-- rna.csv.gz
            |-- cnv.csv.gz
            |-- met.csv.gz
            |-- clin.csv.gz
            |-- readme.md
    |-- model /                         # 存储模型,每个case一个目录
        |-- LGG /
            |-- model.pt
            |-- readme.md
    |-- src /                           # 存储依赖脚本
        |-- features.py                     # 特征筛选
        |-- module.py                       # 构建模型
        |-- explain.py                      # 模型解释
    |-- deepmoi.py                      # 执行脚本
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
pip install requirement.txt
```

## download DeepMOI

```
git clone xxxx.git
```

# parameters

- 




