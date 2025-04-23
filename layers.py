import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class MessagePassing(Module):
    """
    Message Passing Layer
    """
    def __init__(self, input_node_dim, attn_dim = 25):
        super(MessagePassing, self).__init__()
        # self.N = 500
        self.input_node_dim = input_node_dim
        self.output_dim = input_node_dim * 2 + 1
        self.W_attn = Parameter(torch.FloatTensor(self.input_node_dim, attn_dim))
        self.v_attn = Parameter(torch.FloatTensor(attn_dim, 1))
        self.reset_parameters()
            
    def reset_parameters(self):
        node_stdv = 1. / math.sqrt(self.W_attn.size(1))
        self.W_attn.data.uniform_(-node_stdv, node_stdv)
        self.v_attn.data.uniform_(-node_stdv, node_stdv)

    def forward(self, node_features, edge_features, adj):
        # 构建邻居的node features集合
        neighbor_node_features = torch.spmm(adj, node_features)
        # 构建邻居edge fatures的集合
        neighbor_edge_features = torch.spmm(adj, edge_features).to_dense().diagonal().unsqueeze(1) # edge embedding
        return torch.cat((node_features, neighbor_node_features, neighbor_edge_features), dim=1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_node_dim) + ' , ' + str(1) + ' -> ' \
               + str(self.input_node_dim*2+1) + ')'
               
class FeatureUpdate(Module):
    """
    Feature Update Layer
    """
    def __init__(self, input_node_dim, output_node_dim, output_edge_dim, bias=True):
        super(FeatureUpdate, self).__init__()
        self.input_dim = input_node_dim * 2 + 1
        self.output_node_dim = output_node_dim
        self.output_edge_dim = output_edge_dim
        # 初始化模型参数
        self.node_weight = Parameter(torch.FloatTensor(self.input_dim, self.output_node_dim))
        self.edge_weight = Parameter(torch.FloatTensor(self.input_dim, self.output_edge_dim))
        if bias:
            self.bias = bias
            self.node_bias = Parameter(torch.FloatTensor(self.output_node_dim))
            self.edge_bias = Parameter(torch.FloatTensor(self.output_edge_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        node_stdv = 1. / math.sqrt(self.node_weight.size(1))
        self.node_weight.data.uniform_(-node_stdv, node_stdv)
        edge_stdv = 1. / math.sqrt(self.edge_weight.size(1))
        self.edge_weight.data.uniform_(-edge_stdv, edge_stdv)
        if self.bias is not None:
            self.node_bias.data.uniform_(-node_stdv, node_stdv)
            self.edge_bias.data.uniform_(-edge_stdv, edge_stdv)

    def forward(self, message_embedding, adj):
        node_output = torch.mm(message_embedding, self.node_weight)
        edge_output = torch.mm(message_embedding, self.edge_weight)
        edge_output = torch.mul(edge_output, adj.to_dense())
        if self.bias is not None:
            return F.relu(node_output + self.node_bias), F.relu(edge_output + self.edge_bias)
        else:
            return F.relu(node_output), F.relu(edge_output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_node_dim) + ' , ' + str(self.output_edge_dim) + ')'
               
class Readout(Module):
    """
    Readout Layer
    """
    def __init__(self, input_node_dim, output_dim, bias=True):
        super(Readout, self).__init__()
        self.input_dim = input_node_dim * 2 + 1
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, node_features, edge_features, adj):
        # 构建邻居的node features集合
        neighbor_node_features = torch.spmm(adj, node_features)
        # 构建邻居edge fatures的集合
        neighbor_edge_features = torch.spmm(adj, edge_features).to_dense().diagonal().unsqueeze(1) # edge embedding到底要当成向量还是当成一个数值呢
        support = torch.cat((node_features, neighbor_node_features, neighbor_edge_features), dim=1)
        output = torch.mm(support, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'
               
# GCN
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        # output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
