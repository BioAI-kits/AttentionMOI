import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import Set2Set


# class DeepMOI(nn.Module):
#     def __init__(self, in_dim, pathway, add_features=None):
#         """
#         in_dim: == omics' number
#         hidden_dim: == 
#         """
#         super(DeepMOI, self).__init__()
        
#         hidden_dim1 = 256
#         hidden_dim2 = 64
        
#         # GNN-1
#         self.gin_lin1 = torch.nn.Linear(in_dim, in_dim*8)
#         self.conv1 = dglnn.GINConv(self.gin_lin1, 'sum')
        
#         # GNN-2
#         self.gin_lin2 = torch.nn.Linear(in_dim*8, in_dim)
#         self.conv2 = dglnn.GINConv(self.gin_lin2)

#         # GlobalPooling
#         self.sns1 = Set2Set(in_dim, 2, 1)
#         self.sns2 = Set2Set(in_dim, 2, 1)
#         self.sns3 = Set2Set(in_dim, 2, 1)
#         self.ln_gpool = nn.Linear(in_dim*2*3, 1)
        
#         # MLP
#         if add_features == None:
#             self.lin1 = nn.Linear(len(pathway), hidden_dim1)  # not including clinical features 
#         else:
#             add_num = add_features.shape[1]
#             self.lin1 = nn.Linear(len(pathway) + add_num, hidden_dim1)  # including clinical features    
#         self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.lin3 = nn.Linear(hidden_dim2, 1)
        
#         # other args
#         self.pathway = pathway


#     def forward(self, g, h, c=None):
        
#         # subnetwork1: GRL layers
#         h = F.relu(self.conv1(g, h))
#         h = F.relu(self.conv2(g, h))
#         # subnetwork2: patyway layers
#         with g.local_scope():
#             g.ndata['h'] = h
#             # global pooling with Set2Set: output dim = 2*node_dim
#             subgraphs = [dgl.node_subgraph(g, n) for n in self.pathway.values()]
#             graphs_ = dgl.batch(subgraphs)
#             readout1 =  self.sns1(graphs_, graphs_.ndata['h'])
#             readout2 =  self.sns2(graphs_, graphs_.ndata['h'])
#             readout3 =  self.sns3(graphs_, graphs_.ndata['h'])
#             readout = torch.stack([readout1, readout2, readout3], dim=1)
#             readout = readout.reshape(readout.shape[0], -1)
            
#             # compute pathway score
#             x = self.ln_gpool(readout)
            
#             # additional features
#             if c != None:
#                 x = torch.cat([x, c], dim=0)
            
#             # reshape
#             x = x.reshape(1,-1)  # add features
            
#             # linear-1
#             x = nn.Tanh()(self.lin1(x))
            
#             # linear-2
#             x = nn.ReLU()(self.lin2(x))

#             # linear-3
#             logit = nn.Sigmoid()(self.lin3(x))

#             return logit


class DeepMOI(nn.Module):
    def __init__(self, in_dim, pathway, add_features=None):
        """
        in_dim: == omics' number
        hidden_dim: == 
        """
        super(DeepMOI, self).__init__()
        
        hidden_dim1 = 256
        hidden_dim2 = 64
        
        # GNN-1
        self.gin_lin1 = torch.nn.Linear(in_dim, in_dim*3)
        self.conv1 = dglnn.GINConv(self.gin_lin1, 'sum')
        
        # GNN-2
        self.gin_lin2 = torch.nn.Linear(in_dim*3, in_dim*8)
        self.conv2 = dglnn.GINConv(self.gin_lin2)
        
        # GNN-3
        self.gin_lin3 = torch.nn.Linear(in_dim*8, in_dim*16)
        self.conv3 = dglnn.GINConv(self.gin_lin3)
        
        # GNN-4
        self.gin_lin4 = torch.nn.Linear(in_dim*16, in_dim*32)
        self.conv4 = dglnn.GINConv(self.gin_lin4)

        # GlobalPooling
        self.sns1 = Set2Set(in_dim*32, 2, 1)
        self.sns2 = Set2Set(in_dim*32, 2, 1)
        self.sns3 = Set2Set(in_dim*32, 2, 1)
        self.ln_gpool = nn.Linear(in_dim*32*2*3, 1)
        
        # MLP
        if add_features == None:
            self.lin1 = nn.Linear(len(pathway), hidden_dim1)  # not including clinical features 
        else:
            add_num = add_features.shape[1]
            self.lin1 = nn.Linear(len(pathway) + add_num, hidden_dim1)  # including clinical features    
        self.lin2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.lin3 = nn.Linear(hidden_dim2, 1)
        
        # other args
        self.pathway = pathway


    def forward(self, g, h, c=None):
        
        # subnetwork1: GRL layers
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        h = F.relu(self.conv3(g, h))
        h = F.relu(self.conv4(g, h))
        # subnetwork2: patyway layers
        with g.local_scope():
            g.ndata['h'] = h
            # global pooling with Set2Set: output dim = 2*node_dim
            subgraphs = [dgl.node_subgraph(g, n) for n in self.pathway.values()]
            graphs_ = dgl.batch(subgraphs)
            readout1 =  self.sns1(graphs_, graphs_.ndata['h'])
            readout2 =  self.sns2(graphs_, graphs_.ndata['h'])
            readout3 =  self.sns3(graphs_, graphs_.ndata['h'])
            readout = torch.stack([readout1, readout2, readout3], dim=1)
            readout = readout.reshape(readout.shape[0], -1)
            # compute pathway score
            x = self.ln_gpool(readout)
            
            # additional features
            if c != None:
                x = torch.cat([x, c], dim=0)
            
            # reshape
            x = x.reshape(1,-1)  # add features
            
            # linear-1
            x = nn.Tanh()(self.lin1(x))

            # dropout
            x = nn.Dropout(p=0.2)(x)

            # linear-2
            x = nn.ReLU()(self.lin2(x))

            # linear-3
            logit = nn.Sigmoid()(self.lin3(x))

            return logit