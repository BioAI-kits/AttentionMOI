import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import Set2Set


class DeepMOI(nn.Module):
    def __init__(self, in_dim, pathway, clinical_feature=None):
        """
        in_dim: == omics' number
        hidden_dim: == 
        """
        super(DeepMOI, self).__init__()
        # GNN
        self.gin_lin1 = torch.nn.Linear(in_dim, in_dim*2)
        self.conv1 = dglnn.GINConv(self.gin_lin1, 'sum')

        self.gin_lin2 = torch.nn.Linear(in_dim*2, in_dim)
        self.conv2 = dglnn.GINConv(self.gin_lin2)

        # GlobalPooling
        self.sns1 = Set2Set(3, 2, 1)
        self.sns2 = Set2Set(3, 2, 1)
        self.sns3 = Set2Set(3, 2, 1)
        
        # MLP
        self.lin1 = nn.Linear(len(pathway)*in_dim*2, len(pathway))
        if clinical_feature == None:
            self.lin2 = nn.Linear(len(pathway), 1)  # not including clinical features 
        else:
            clinical_feature_num = clinical_feature.shape[1]
            self.lin2 = nn.Linear(len(pathway) + clinical_feature_num, 128)  # including clinical features
        self.lin3 = nn.Linear(128, 1)
        self.pathway = pathway

        

    def forward(self, g, h, c=None):
        # subnetwork1: GRL layers
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))

        # subnetwork2: patyway layers
        with g.local_scope():
            g.ndata['h'] = h

            # global pooling with Set2Set: output dim = 2*node_dim
            subgraphs = [dgl.node_subgraph(g, n) for n in self.pathway.values()]
            graphs_ = dgl.batch(subgraphs)
            readout1 =  self.sns1(graphs_, graphs_.ndata['h'])
            readout1 = readout1.reshape(1,-1).squeeze(0)
            readout2 =  self.sns2(graphs_, graphs_.ndata['h'])
            readout2 = readout2.reshape(1,-1).squeeze(0)
            readout3 =  self.sns3(graphs_, graphs_.ndata['h'])
            readout3 = readout3.reshape(1,-1).squeeze(0)
            
            # linear-1
            x = nn.Tanh()(self.lin1(readout1))
            
            if c != None:
                x = torch.cat([x, c], dim=0)
            
            # linear-2
            x = nn.ReLU()(self.lin2(x))

            # linear-3
            logit = nn.Sigmoid()(self.lin3(x))

            return logit