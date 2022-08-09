import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn


class DeepMOI(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(DeepMOI, self).__init__()
        self.lin1 = nn.Linear(in_feat, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
#         x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, 0.5)
        
        x = self.lin2(x)
        x = torch.relu(x)
#         x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, 0.5)
        
        x = self.lin3(x)
        x = torch.relu(x)
#         x = F.leaky_relu(x, 0.25)
        x = F.dropout(x, 0.5)
        
#         x = nn.Dropout(p=0.4)(x)
        x = self.lin4(x)
        logit = torch.sigmoid(x)
        
        return logit


# class DeepMOI(nn.Module):
#     def __init__(self, in_feats, out_feats):
#         super().__init__()
        
#         self.conv1 = dglnn.GATConv(
#             in_feats=in_feats, out_feats=256, num_heads=1)
        
#         self.conv2 = dglnn.GATConv(
#             in_feats=256, out_feats=128, num_heads=1)
        
#         self.conv3 = dglnn.GATConv(
#             in_feats=128, out_feats=64, num_heads=1)
        
#         self.conv4 = dglnn.GATConv(
#             in_feats=64, out_feats=1, num_heads=1)
        

#     def forward(self, graph, h):
#         with graph.local_scope():
#             h = self.conv1(graph, h)
#             h = h[:,0,:]
#             h = torch.tanh(h)
#             h = nn.Dropout(p=0.2)(h)
            
#             h = self.conv2(graph, h)
#             h = h[:,0,:]
#             h = F.relu(h)
#             h = nn.Dropout(p=0.2)(h)
            
#             h = self.conv3(graph, h)
#             h = h[:,0,:]
#             h = F.relu(h)
#             h = nn.Dropout(p=0.2)(h)
            
            
#             h = self.conv4(graph, h)
#             h = h[:,0,:]
#             h = torch.sigmoid(h)
            
#             return h

