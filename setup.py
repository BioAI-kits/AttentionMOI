from setuptools import setup, find_packages


install_packages = [
    'captum>=0.4.1',
    'mygene>=3.2.2',
    'openpyxl>=3.0.9',
    'packaging>=21.3',
    'pandas>=1.2.5',
    'pandocfilters>=1.5.0',
    'seaborn>=0.11.2',
    'torch==1.13.1',
    'scikit-learn>=1.2.2',
    'numpy>=1.23.5',
    'matplotlib>=3.6.2',
    'xgboost>=1.7.4',
    'livelossplot', 
    'tensorboardX',
    'tqdm',
    'ipython',
]

setup(
    # 应用名
    name='AttentionMOI',
    # 作者名
    author='Billy',
    # 作者邮箱
    author_email='liangbilin0324@163.com',
    # 版本号
    version='0.1.2',
    # 要求python版本
    python_requires=">=3.9.*",
    # 找到本目录下的所有python包
    packages=find_packages(),
    # 自动安装依赖
    install_requires=install_packages,
    dependency_links=[
        "https://pypi.org/simple/",
        "https://download.pytorch.org/whl/cpu#egg=torch",
        ],
    # 程序网站
    url='https://github.com/BioAI-kits/AttentionMOI',
    # 程序简单描述
    description="A Denoised Multi-omics Integration Framework for Cancer Subtype Classification and Survival Prediction.",
    # 开源许可
    license='Apache License 2.0',
    # 包含的数据
    data_files=['AttentionMOI/example/cnv.csv.gz', 'AttentionMOI/example/met.csv.gz', 'AttentionMOI/example/rna.csv.gz', 'AttentionMOI/example/label.csv'],
    # 命令行
    entry_points={
        'console_scripts': ['moi = AttentionMOI.moi:run_main',
                            ],
    },
)


