import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import MessagePassing, FeatureUpdate, Readout, GraphConvolution
import numpy as np
from gpytorch import inv_matmul, matmul, logdet

class MP_model(nn.Module):
    def __init__(self, input_node_dim, node_output_dim, node_hidden_dim, edge_hidden_dim, dropout):
        super(MP_model, self).__init__()
        self.num_layer = len(node_hidden_dim)
        # 定义MessagePassing
        self.M = nn.ModuleList([MessagePassing(input_node_dim)])
        # 定义FeatureUpdate
        self.U = nn.ModuleList([FeatureUpdate(input_node_dim, node_hidden_dim[0], edge_hidden_dim)])
        
        for l in range(self.num_layer-1):
            self.M.append(MessagePassing(node_hidden_dim[l]))
            self.U.append(FeatureUpdate(node_hidden_dim[l], node_hidden_dim[l+1], edge_hidden_dim))
        
        # 定义Readout
        self.R = Readout(node_hidden_dim[self.num_layer-1], node_output_dim)
        
        # 定义dropout
        self.dropout = dropout

    def forward(self, node_features, edge_features, adj): # node_features 是 Nx2，edge_features 是 NxN
        for l in range(self.num_layer):
            H = self.M[l](node_features, edge_features, adj)
            node_features, edge_features = self.U[l](H, adj)
            # dropout
            node_features = F.dropout(node_features, self.dropout, training=self.training)
            edge_features = F.dropout(edge_features, self.dropout, training=self.training)
            
        node_output = self.R(node_features, edge_features, adj)
        return node_output
    
    def extra_repr(self):
        return "MP_model"
    
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid[0])

        # self.gc2 = GraphConvolution(nhid[0], nhid[1])
        # self.gc3 = GraphConvolution(nhid[1], nhid[2])
        # self.gc4 = GraphConvolution(nhid[2], nhid[3])

        self.gc5 = GraphConvolution(nhid[0], nout)
        self.dropout = dropout

    def forward(self, x, adj):
        # Generate Initial node features
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        # x = F.relu(self.gc2(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc3(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.relu(self.gc4(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)

        x = self.gc5(x, adj)
        return x
    
    def extra_repr(self):
        return "GCN"