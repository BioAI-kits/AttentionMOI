import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
# from dgl.nn import Set2Set

from torch_geometric.nn import SAGEConv, SAGPooling, Set2Set, GraphNorm, global_sort_pool, GlobalAttention
from torch_geometric.utils import add_self_loops, subgraph
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

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
#         # h = nn.Dropout(p=0.001)(h)

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
            
#             x = nn.Dropout(p=0.6)(x)

#             # linear-2
#             x = nn.ReLU()(self.lin2(x))

#             # linear-3
#             logit = nn.Sigmoid()(self.lin3(x))

#             return logit


# class DeepMOI(nn.Module):
#     def __init__(self, in_dim, pathway, add_features=None):
#         """
#         in_dim: == omics' number
#         hidden_dim: == 
#         """
#         super(DeepMOI, self).__init__()
#         # GNN-1
#         self.conv1 = SAGEConv(in_dim, in_dim*8)
#         self.pool1 = SAGPooling(in_dim*8, ratio=0.8)
#         self.readout1 = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
#         # GNN-2
#         self.conv2 = SAGEConv(in_dim*8, in_dim*8)
#         self.pool2 = SAGPooling(in_dim*8, ratio=0.8)
#         self.readout2 = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
#         # GNN-3
#         self.conv3 = SAGEConv(in_dim*8, in_dim*8)
#         self.pool3 = SAGPooling(in_dim*8, ratio=0.8)
#         self.readout3 = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
#         # MLP
#         self.mlp = nn.Sequential(
#                                  nn.Linear(in_dim*8*3, 48),
#                                  nn.Tanh(),
#                                  nn.Dropout(p=0.4),
#                                  nn.Linear(48, 16),
#                                  nn.Tanh(),
#                                  nn.Linear(16,1),
#                                  nn.Sigmoid()
#                                 )
#         self.norm = GraphNorm(in_dim*8)
    
#     def forward(self, g, h, c=None):
#         edge_index = g.edge_index
#         edge_index,_ = add_self_loops(edge_index=edge_index)
#         x = h
#         # GNN-1
#         x = torch.tanh(self.conv1(x, edge_index))
#         x, edge_index, _, _, _, _ = self.pool1(x, edge_index, None, None)
#         x1 = self.readout1(x, None)

#         # GNN-2
#         x = self.norm(x)
#         x = torch.tanh(self.conv2(x, edge_index))
#         x, edge_index, _, _, _, _ = self.pool2(x, edge_index, None, None)
#         x2 = self.readout2(x)

#         # GNN-3
#         x = self.norm(x)
#         x = torch.tanh(self.conv3(x, edge_index))
#         x, edge_index, _, _, _, _ = self.pool3(x, edge_index, None, None)
#         x3 = self.readout3(x)

#         # concat readout
#         readout = torch.cat([x1, x2, x3], dim=1)

#         # MLP
#         logit = torch.sigmoid(self.mlp(readout))

#         return logit

class DeepMOI(nn.Module):
    def __init__(self, in_dim, pathway, add_features=None):
        """
        in_dim: == omics' number
        hidden_dim: == 
        """
        super(DeepMOI, self).__init__()
        self.pathway = pathway
        
        # GNN-1
        self.conv_a = SAGEConv(in_dim, in_dim*8)
        self.readout_a = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
        
        # GNN-2
        self.conv_b = SAGEConv(in_dim*8, in_dim*8)
        self.readout_b = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
        
        # GNN-3
        self.conv_c = SAGEConv(in_dim*8, in_dim*8)
        self.readout_c = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
        
        
        
        # GNN-1
        self.conv1 = SAGEConv(in_dim*8, in_dim*8)
        self.pool1 = SAGPooling(in_dim*8, ratio=0.8)
        self.readout1 = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
        # GNN-2
        self.conv2 = SAGEConv(in_dim*8, in_dim*8)
        self.pool2 = SAGPooling(in_dim*8, ratio=0.8)
        self.readout2 = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
        # GNN-3
        self.conv3 = SAGEConv(in_dim*8, in_dim*8)
        self.pool3 = SAGPooling(in_dim*8, ratio=0.8)
        self.readout3 = GlobalAttention(gate_nn=nn.Linear(in_dim*8, 1))
        
        # lin
        self.lin = nn.Linear(in_dim*8*3, 1)
        
        # MLP
        
        self.mlp = nn.Sequential(
                                 nn.Linear(len(self.pathway.pathway.unique()) + 1, 48),
                                 nn.Tanh(),
                                 nn.Dropout(p=0.4),
                                 nn.Linear(48, 16),
                                 nn.Tanh(),
                                 nn.Linear(16,1),
                                 nn.Sigmoid()
                                )
        self.norm = GraphNorm(in_dim*8)
        
        
    
    def forward(self, g, h, c=None):
        edge_index = g.edge_index
        edge_index,_ = add_self_loops(edge_index=edge_index)
        x = h
        # GNN-1
        x = torch.tanh(self.conv_a(x, edge_index))
        x1 = self.readout_a(x, None)

        # GNN-2
        x = self.norm(x)
        x = torch.tanh(self.conv_b(x, edge_index))
        x2 = self.readout_b(x)

        # GNN-3
        x = self.norm(x)
        x = torch.tanh(self.conv_c(x, edge_index))
        x3 = self.readout_c(x)

        # concat readout
        readout1 = torch.cat([x1, x2, x3], dim=1)
        
        ###
        batch_size = len(self.pathway.pathway.unique())
        subgraphs = []
        for path, group in self.pathway.groupby('pathway'):
            nodes = list(set(group.src.to_list() + group.dest.to_list()))
            sub_edge_idx,_ = subgraph(subset=nodes, edge_index=edge_index)
            subgraphs.append(Data(x=x, edge_index=sub_edge_idx))
        dataset = DataLoader(subgraphs, batch_size=batch_size)
        for dat in dataset:
            x, edge_index, batch = dat.x, dat.edge_index, dat.batch
            # GNN-1
            x = torch.tanh(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            x1 = self.readout1(x, batch)
            # GNN-2
            x = torch.tanh(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            x2 = self.readout2(x, batch)
            # GNN-3
            x = torch.tanh(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x3 = self.readout3(x, batch)
            readout2 = torch.cat([x1, x2, x3], dim=1)
            
        readout = torch.cat([readout1, readout2], dim=0)
        readout = torch.tanh(self.lin(readout).T)
        # MLP
        logit = torch.sigmoid(self.mlp(readout))

        return logit