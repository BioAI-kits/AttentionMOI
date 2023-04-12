import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMOI(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DeepMOI, self).__init__()
        self.lin1 = nn.Linear(in_feat, 256)
        self.lin2 = nn.Linear(256, 64)
        self.lin3 = nn.Linear(64, out_feat)
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = F.dropout(x, 0.5)
        
        x = self.lin2(x)
        x = torch.relu(x)
        x = F.dropout(x, 0.5)
        
        x = self.lin3(x)
        
        return x


# class Net(nn.Module):
#     def __init__(self, dim_dna, dim_rna, dim_out):
#         """
#         dim_dna: DNA 组学数据特征维度
#         dim_rna: RNA 组学数据特征维度
#         dim_out: 分类任务的类别数量
#         """
#         super().__init__()
#         # 用于提取DNA组学特征的模块
#         self.dna = nn.Sequential(
#             nn.Linear(dim_dna, 128),
#             nn.BatchNorm1d(num_features=128),
#             nn.Linear(128, dim_dna),
#         )
#         # 用于提取RNA组学特征的模块
#         self.rna = nn.Sequential(
#             nn.Linear(dim_rna, 128),
#             nn.BatchNorm1d(num_features=128),
#             nn.Linear(128, dim_rna),
#         )
#         # 用于分类任务的模块
#         self.mlp = nn.Sequential(
#             nn.Linear(dim_dna + dim_rna, 128),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.BatchNorm1d(num_features=128),
#             nn.Linear(128, dim_out)
#         )
#         self.norm = nn.BatchNorm1d(num_features=dim_dna + dim_rna)

#     def forward(self, data_dna, data_rna):
#         # 提取DNA特征
#         feat_dna = self.dna(data_dna)
#         # 提取RNA特征
#         feat_rna = self.rna(data_rna)
#         # 合并特征
#         h = torch.cat([feat_dna, feat_rna], dim=1)
#         h = self.norm(h)
#         h = h.squeeze(-1)
#         # 分类
#         out = self.mlp(h)
#         return out


class Net(nn.Module):
    def __init__(self, dim_dna, dim_rna, dim_out):
        """
        dim_dna: DNA 组学数据特征维度
        dim_rna: RNA 组学数据特征维度
        dim_out: 分类任务的类别数量
        """
        super().__init__()
        # 用于提取DNA组学特征的模块
        self.dna = nn.Sequential(
            nn.BatchNorm1d(num_features=dim_dna),
            nn.TransformerEncoderLayer(d_model=dim_dna, nhead=1, dim_feedforward=128),
        )
        # 用于提取RNA组学特征的模块
        self.rna = nn.Sequential(
            nn.BatchNorm1d(num_features=dim_rna),
            nn.TransformerEncoderLayer(d_model=dim_rna, nhead=1, dim_feedforward=128),
        )
        # 用于分类任务的模块
        self.mlp = nn.Sequential(
            nn.Linear(dim_dna + dim_rna, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, dim_out)
        )
        self.norm = nn.BatchNorm1d(num_features=dim_dna + dim_rna)

    def forward(self, data_dna, data_rna):
        # 提取DNA特征
        feat_dna = self.dna(data_dna)
        # 提取RNA特征
        feat_rna = self.rna(data_rna)
        # 合并特征
        h = torch.cat([feat_dna, feat_rna], dim=1)
        h = self.norm(h)
        h = h.squeeze(-1)
        # 分类
        out = self.mlp(h)
        return out
