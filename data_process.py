import numpy as np
import torch
from scipy.io import savemat, loadmat
import scipy.sparse as sp

def adjacent_normalize(mx):
    """"D^(-0.5)*A*D^(-0.5)"""
    rowsum = np.array(mx.sum(1))
    r_inv_half = np.power(rowsum, -0.5).flatten()
    r_inv_half[np.isinf(r_inv_half)] = 0.
    r_mat_inv = sp.diags(r_inv_half)
    mx = mx.dot(r_mat_inv).transpose().dot(r_mat_inv)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_data(data_path, threshold, percent_train, percent_val, percent_test): # idx_train, idx_val, idx_test都是range
    # Load the dataset
    m = loadmat(data_path)
    Range_Mat = m["range"]  # Range = Distance + noise
    Dist_Mat = m["dist"]
    labels = m["labels"]
    N_anchors = m["N_anchors"][0][0]
    N_agents = m["N_agents"][0][0]
    # Size information
    num_sample = Range_Mat.shape[0]

    # 瓜分训练集，验证集以及测试集
    idx_train = torch.LongTensor(range(int(num_sample * percent_train)))
    idx_val = torch.LongTensor(range(int(num_sample * percent_train), int(num_sample * (percent_train + percent_val))))
    idx_test = torch.LongTensor(range(int(num_sample * (percent_train + percent_val)), num_sample))
    # idx_train = torch.LongTensor(range(num_sample))
    # idx_val = torch.LongTensor(range(1))
    # idx_test = torch.LongTensor(range(1))
    
    # 需要返回的数据
    edge_range_all = [] # 测出来的edge_features，可能带noise
    edge_dist_all = [] # 不带noise的edge_features
    rownorm_adj_all = [] # 非环且行归一化的adj，用于Message Passing
    norm_adj_all = [] # 非自环且归一化的adj，用于residual correlation
    self_rownorm_adj_all = [] # 自环且归一化的adj，用于GCN
    edge_dist_anchors = Dist_Mat[0][range(N_anchors), :][:, range(N_anchors)] # anchors的距离矩阵，不带noise
    edge_dist_nodes = Range_Mat[0][:, range(N_anchors)] # agents的距离矩阵，带noise
    
    for k in range(num_sample):
        Range = Range_Mat[k].copy()
        Dist = Dist_Mat[k].copy()
        
        # 应用node relationship threshold
        Range[Range > threshold] = 0
        Dist[Range > threshold] = 0
        
        # 获取带noise跟不带noise的edge_features
        edge_range = Range.copy()
        edge_dist = Dist.copy()
        edge_range = sp.csr_matrix(edge_range, dtype=np.float64)
        edge_dist = sp.csr_matrix(edge_dist, dtype=np.float64)
        edge_range = normalize(edge_range)
        edge_dist = normalize(edge_dist)
        
        # 获取不同的adjacency matrices
        # rownorm_adj
        rownorm_adj = Range.copy()
        rownorm_adj[rownorm_adj > 0] = 1
        rownorm_adj = sp.csr_matrix(rownorm_adj, dtype=np.float64)
        rownorm_adj = normalize(rownorm_adj)
        # norm_adj
        norm_adj = Range.copy()
        norm_adj[norm_adj > 0] = 1
        norm_adj = sp.csr_matrix(norm_adj, dtype=np.float64)
        norm_adj = adjacent_normalize(norm_adj)
        # self_rownorm_adj
        self_rownorm_adj = Range.copy()
        self_rownorm_adj[self_rownorm_adj > 0] = 1
        self_rownorm_adj = sp.csr_matrix(self_rownorm_adj, dtype=np.float64)
        self_rownorm_adj = normalize(self_rownorm_adj + sp.eye(self_rownorm_adj.shape[0]))

        # 储存数据
        edge_range_all.append(sparse_mx_to_torch_sparse_tensor(edge_range))
        edge_dist_all.append(sparse_mx_to_torch_sparse_tensor(edge_dist))
        rownorm_adj_all.append(sparse_mx_to_torch_sparse_tensor(rownorm_adj))
        norm_adj_all.append(sparse_mx_to_torch_sparse_tensor(norm_adj))
        self_rownorm_adj_all.append(sparse_mx_to_torch_sparse_tensor(self_rownorm_adj))
    
    labels_all = torch.FloatTensor(labels)
    edge_dist_anchors = torch.FloatTensor(edge_dist_anchors)
    edge_dist_nodes = torch.FloatTensor(edge_dist_nodes)
    
    return num_sample, N_anchors, N_agents, idx_train, idx_val, idx_test, edge_range_all, edge_dist_all, rownorm_adj_all, norm_adj_all, self_rownorm_adj_all, labels_all, edge_dist_anchors, edge_dist_nodes