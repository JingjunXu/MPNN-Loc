import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from layers import GraphConvolution
import numpy as np
from gpytorch import inv_matmul, matmul, logdet


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

class C_GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nout, dropout):
        super(C_GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nout)
        
        # self.coeff = Parameter(torch.FloatTensor(2, 1))
        # self.coeff.data = torch.tensor([[1.0], [1.0]])
        
        self.dropout = dropout
    
    def forward(self, x, adj, labels, idx_anchors, idx_agents, alpha, beta):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = self.lp_refine(idx_agents, idx_anchors, labels, x, adj, alpha, beta)
        return x
        
    def get_Gamma(self, S, alpha, beta):
        device = S.device
        # return torch.exp(self.coeff[1]) * torch.eye(S.size(0), device=device) - torch.exp(self.coeff[1]) * torch.tanh(self.coeff[0]) * S
        return torch.exp(beta) * torch.eye(S.size(0), device=device) - torch.exp(beta) * torch.tanh(alpha) * S

    def interpolate(self, idx_train, idx_test, res_pred_train, Gamma):
        device = Gamma.device
        
        idx_train = idx_train.to(torch.long).to(device)
        idx_test = idx_test.to(torch.long).to(device)
        idx = np.arange(Gamma.shape[0])
        idx_val = np.setdiff1d(idx, np.concatenate((idx_train.cpu().numpy(), idx_test.cpu().numpy())))
        idx_test_val = np.concatenate((idx_test.cpu().numpy(), idx_val))
        test_val_Gamma = Gamma[idx_test_val, :][:, idx_test_val]
        
        res_pred_test = matmul(torch.inverse(test_val_Gamma), -matmul(Gamma[idx_test_val, :][:, idx_train], res_pred_train))
        return res_pred_test[:len(idx_test)]

    def lp_refine(self, idx_agents, idx_anchors, labels, output, S, alpha, beta):
        Gamma = self.get_Gamma(S, alpha, beta)

        pred_train = output[idx_anchors]
        pred_test = output[idx_agents]
        res_pred_train = labels[idx_anchors] - output[idx_anchors]
        
        refined_test = pred_test + self.interpolate(idx_anchors, idx_agents, res_pred_train, Gamma)

        return torch.cat((pred_train, refined_test), dim=0)