import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


# class DeepMOI(nn.Module):
#     def __init__(self, in_feat, out_feat):
#         super(DeepMOI, self).__init__()
#         self.lin1 = nn.Linear(in_feat, 256)
#         self.lin2 = nn.Linear(256, 64)
# #         self.lin3 = nn.Linear(128, 64)
#         self.lin4 = nn.Linear(64, 1)
        
#     def forward(self, x):
#         x = self.lin1(x)
#         x = torch.relu(x)
# #         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, 0.5)
        
#         x = self.lin2(x)
#         x = torch.relu(x)
# #         x = F.leaky_relu(x, 0.25)
#         x = F.dropout(x, 0.5)
        
# #         x = self.lin3(x)
# #         x = torch.relu(x)
# # #         x = F.leaky_relu(x, 0.25)
# #         x = F.dropout(x, 0.5)
        
# #         x = nn.Dropout(p=0.4)(x)
#         x = self.lin4(x)
#         logit = torch.sigmoid(x)
        
#         return logit


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