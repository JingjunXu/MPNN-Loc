from __future__ import division
from __future__ import print_function

import time
import datetime
import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from gpytorch import matmul, logdet
from sklearn.metrics import r2_score

from data_process import load_data
from models import MP_model, GCN

import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=3e-5, # 探索 1e-2, 5e-3, 1e-3, 5e-4, 1e-4
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--node_hidden', type=int, default=[200],
                    help='Number of node hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
threshold = 0.2
seed = 42
percent_train = 0.7
percent_val = 0.1
percent_test = 0.2

# Fix the random seed
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)

# Load data
data_path = "./Networks/Uniform_100samples_25anchors_1875agents_00625Variance.mat"
num_sample, N_anchors, N_agents, idx_train, idx_val, idx_test, edge_range_all, edge_dist_all, rownorm_adj_all, norm_adj_all, self_rownorm_adj_all, labels_all, edge_dist_anchors, edge_dist_nodes = load_data(data_path, threshold, percent_train, percent_val, percent_test)

# 确定anchors跟agents的index
N = N_anchors + N_agents
idx_anchors = range(N_anchors)
idx_agents = range(N_anchors, N_anchors + N_agents)

# Model
# model = GCN(nfeat=N,
#             nhid=args.node_hidden,
#             nout=2,
#             dropout=args.dropout)

model = MP_model(input_node_dim=2, # 2个本节点的node features，2个另一个节点的node features，以及它们之间边的edge feature
                node_output_dim=2,
                node_hidden_dim=args.node_hidden,
                edge_hidden_dim=N, 
                dropout=args.dropout)

# Transfer data to cuda
if args.cuda:
    model.cuda()
    edge_dist_anchors = edge_dist_anchors.cuda()
    edge_dist_nodes = edge_dist_nodes.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_anchors = torch.LongTensor(idx_anchors).cuda()
    idx_agents = torch.LongTensor(idx_agents).cuda()
    labels_all = labels_all.cuda()
    for k in range(100):
        edge_dist_all[k] = edge_dist_all[k].cuda()
        edge_range_all[k] = edge_range_all[k].cuda()
        labels_all[k] = labels_all[k].cuda()
        # edge_features_all[k] = edge_features_all[k].cuda()
        # node_features_all[k] = node_features_all[k].cuda()
        rownorm_adj_all[k] = rownorm_adj_all[k].cuda()
        norm_adj_all[k] = norm_adj_all[k].cuda()
        self_rownorm_adj_all[k] = self_rownorm_adj_all[k].cuda()
    
# Optimizer
optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

# Refined Functions
def interpolate(idx_train, idx_test, res_pred_train, Gamma):
    device = Gamma.device
    
    idx_train = idx_train.to(torch.long).to(device)
    idx_test = idx_test.to(torch.long).to(device)
    idx = np.arange(Gamma.shape[0])
    idx_val = np.setdiff1d(idx, np.concatenate((idx_train.cpu().numpy(), idx_test.cpu().numpy())))
    idx_test_val = np.concatenate((idx_test.cpu().numpy(), idx_val))
    test_val_Gamma = Gamma[idx_test_val, :][:, idx_test_val]
    
    res_pred_test = matmul(torch.inverse(test_val_Gamma), -matmul(Gamma[idx_test_val, :][:, idx_train], res_pred_train))
    return res_pred_test[:len(idx_test)]

def lp_refine(idx_test, idx_train, labels, output, adj, alpha0=1., beta0=1., alpha1=1., beta1=1.):
    device = adj.device
    Gamma0 = (torch.eye(adj.size(0), device=device) - alpha0*adj)*np.exp(beta0)
    Gamma1 = (torch.eye(adj.size(0), device=device) - alpha1*adj)*np.exp(beta1)
    
    pred_train = output[idx_train]
    pred_test = output[idx_test]
    res_pred_train0 = labels[idx_train][:, 0] - output[idx_train][:, 0]
    res_pred_train1 = labels[idx_train][:, 1] - output[idx_train][:, 1]
    
    residual_test0 = interpolate(idx_train, idx_test, res_pred_train0, Gamma0)
    residual_test1 = interpolate(idx_train, idx_test, res_pred_train1, Gamma1)
    
    refined_test = pred_test + torch.stack((residual_test0, residual_test1), dim=1)

    return refined_test

# node_features and edge_features
edge_features_all = edge_range_all
node_features_all = []
def get_node_features(anchors_labels, edge_features, adj):
    epsilon = 1e-5
    node_features = torch.zeros(N, 2)
    if args.cuda:
        node_features = node_features.cuda()
    
    # set agents' features the same as their nearest anchors
    agents_weights = edge_features[idx_agents][:, idx_anchors]
    agents_weights = agents_weights / (agents_weights.sum(dim=1, keepdim=True)+epsilon)
    min_non_zero_values = torch.where(agents_weights != 0, agents_weights, torch.tensor(float('inf'))).amin(dim=1, keepdim=True)
    agents_weights = torch.where(agents_weights == min_non_zero_values, torch.tensor(1.0), torch.where(agents_weights != 0, torch.tensor(0.0), agents_weights))
    # set outlier agents' features at the middle of the network
    all_zero_rows = (agents_weights == 0).all(dim=1)
    agents_weights[all_zero_rows] = 1. / agents_weights.size(1)

    node_features[idx_agents] = torch.mm(agents_weights, anchors_labels)

    node_features[idx_anchors] = anchors_labels
    return node_features

for k in range(num_sample):
    node_features = get_node_features(labels_all[k][idx_anchors], edge_features_all[k].to_dense(), rownorm_adj_all[k])
    node_features_all.append(node_features)

# Subsidiary Functions for Training Loss
# Diatance
def compute_distance_matrix(A, B):
    A_expanded = A.unsqueeze(1)
    B_expanded = B.unsqueeze(0)

    dist_sq = torch.sum((A_expanded - B_expanded) ** 2, dim=2)
    dist = torch.sqrt(dist_sq)
    return dist

# Criterion
# RMSE loss function
loss_fun = torch.nn.MSELoss()
    
# R2 Function
def R2(outputs, labels):
    outputs = outputs.cpu().detach().numpy().reshape(-1)
    labels = labels.cpu().detach().numpy().reshape(-1)
    return r2_score(labels, outputs)

# Train
def train(node_features, edge_features, adj, labels, epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(node_features, edge_features, adj)
    loss_train = loss_fun(output, labels)
    loss_train.backward()
    optimizer.step()

    loss_anchors = torch.sqrt(loss_fun(output[idx_anchors], labels[idx_anchors]))
    loss = torch.sqrt(loss_fun(output[idx_agents], labels[idx_agents]))
    r2 = R2(output[idx_agents], labels[idx_agents])
    
    # Evaluate the model
    if not args.fastmode:
        model.eval()
        output = model(node_features, edge_features, adj)
    
    loss_val = torch.sqrt(loss_fun(output[idx_agents], labels[idx_agents]))
    r2_val = R2(output[idx_agents], labels[idx_agents])
    return loss_anchors, loss, r2, loss_val, r2_val

# Test
def test(node_features, edge_features, rownorm_adj, norm_adj, labels, epoch):
    model.eval()
    output = model(node_features, edge_features, rownorm_adj)
    
    # refinement
    prediction_raw = lp_refine(idx_agents, idx_anchors, labels, output, norm_adj)
    
    loss = torch.sqrt(loss_fun(prediction_raw, labels[idx_agents]))
    r2_test_refine_raw = R2(prediction_raw, labels[idx_agents])
  
    return loss, r2_test_refine_raw

# Variables declaration
loss_anchors_total = np.zeros(args.epochs)
loss_train_total = np.zeros(args.epochs) # RMSE
r2_train_total = np.zeros(args.epochs) # Check the Correlation
loss_val_total = np.zeros(args.epochs)
r2_val_total = np.zeros(args.epochs)
loss_test_total = np.zeros(args.epochs)
r2_test_total = np.zeros(args.epochs)
alpha = np.zeros(args.epochs) # Trainable variable
beta = np.zeros(args.epochs) # Trainable variable

print("\nTraining!\n")
for epoch in range(args.epochs):
    # Train the model
    t_epoch = time.time()
    loss_anchors = []
    loss_train = []
    r2_train = []
    if not args.fastmode:
        loss_val = []
        r2_val = []
    for batch in idx_train:
        loss_anchors_tem, loss_train_tem, r2_train_tem, loss_val_tem, r2_val_tem = train(node_features_all[batch], edge_features_all[batch], rownorm_adj_all[batch], labels_all[batch], epoch)
        loss_anchors.append(loss_anchors_tem)
        loss_train.append(loss_train_tem)
        r2_train.append(r2_train_tem)
        if not args.fastmode:
            loss_val.append(loss_val_tem)
            r2_val.append(r2_val_tem)
    loss_anchors_total[epoch] = sum(loss_anchors) / len(loss_anchors)
    loss_train_total[epoch] = sum(loss_train) / len(loss_train)
    r2_train_total[epoch] = sum(r2_train) / len(r2_train)
    if not args.fastmode:
        loss_val_total[epoch] = sum(loss_val) / len(loss_val)
        r2_val_total[epoch] = sum(r2_val) / len(r2_val)
        
    # Test the model
    loss_test = []
    r2_test = []
    for batch in idx_test:
        loss_test_tem, r2_test_tem = test(node_features_all[batch], edge_features_all[batch], rownorm_adj_all[batch], norm_adj_all[batch], labels_all[batch], epoch)
        loss_test.append(loss_test_tem)
        r2_test.append(r2_test_tem)
    loss_test_total[epoch] = sum(loss_test) / len(loss_test)
    r2_test_total[epoch] = sum(r2_test) / len(r2_test)
    
    print('Epoch: {:04d}'.format(epoch + 1),
            'loss_anchors (RMSE): {:.4f}'.format(loss_anchors_total[epoch].item()),
            'loss_train (RMSE): {:.4f}'.format(loss_train_total[epoch].item()),
            'r2_train: {:.4f}'.format(r2_train_total[epoch].item()),
            'loss_val (RMSE): {:.4f}'.format(loss_val_total[epoch].item()),
            'r2_val: {:.4f}'.format(r2_val_total[epoch].item()),
            'loss_test (RMSE): {:.4f}'.format(loss_test_total[epoch].item()),
            'r2_test: {:.4f}'.format(r2_test_total[epoch].item()),
            'time: {:.4f}s'.format(time.time() - t_epoch)
            )

# Save model's parameters
# torch.save(model.state_dict(), 'model_weights.pth')

loss_anchors_final = sum(loss_anchors_total) / args.epochs
loss_train_final = sum(loss_train_total) / args.epochs
r2_train_final = sum(r2_train_total) / args.epochs
loss_val_final = sum(loss_val_total) / args.epochs
r2_val_final = sum(r2_val_total) / args.epochs
loss_test_final = sum(loss_test_total) / args.epochs
r2_test_final = sum(r2_test_total) / args.epochs

print("=====================================\n")
print("Averaged Train results:", 
      "loss= {:.4f} (RMSE)".format(loss_train_final), 
      "r2= {:.4f}\n".format(r2_train_final), 
      "Averaged Evaluate results:", 
      "loss= {:.4f} (RMSE)".format(loss_val_final), 
      "r2= {:.4f}\n".format(r2_val_final), 
      "Averaged Test results:", 
      "loss= {:.4f} (RMSE)".format(loss_test_final), 
      "r2= {:.4f}\n".format(r2_test_final), 
    )

best_idx = np.argmin(loss_train_total)
print("\nBest Train results:", 
      "loss= {:.4f} (RMSE)".format(loss_train_total[best_idx]), 
      "r2= {:.4f}\n".format(r2_train_total[best_idx]), 
      "Best Evaluate results:", 
      "loss= {:.4f} (RMSE)".format(loss_val_total[best_idx]), 
      "r2= {:.4f}\n".format(r2_val_total[best_idx]), 
      "Best Test results:", 
      "loss= {:.4f} (RMSE)".format(loss_test_total[best_idx]), 
      "r2= {:.4f}\n".format(r2_test_total[best_idx]), 
    )

nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')  # Get the Now time

file_handle = open('result.txt', mode='a')
file_handle.write('=====================================\n')
file_handle.write(nowTime + '\n')
print(model, file=file_handle)
file_handle.write("Dataset: " + str(data_path) + "\n")
file_handle.write("Threshold: " + str(threshold) + "\n")
# Traing Setting
file_handle.write("Traing Setting:\n")
file_handle.write("Random Seed: " + str(args.seed) + "\n")
file_handle.write("Epoches: " + str(args.epochs) + "\n")
file_handle.write("Learning Rate: " + str(args.lr) + "\n")
file_handle.write("Weight Decay: " + str(args.weight_decay) + "\n")
file_handle.write("Hidden node numbers: " + str(args.node_hidden) + "\n")
file_handle.write("Dropout: " + str(args.dropout) + "\n")
# Results
file_handle.write('Average Results:' + '\n')
file_handle.write('loss_train (RMSE): ' + str(loss_train_final) + 'r2_train: ' + str(r2_train_final) + '\n')
file_handle.write('loss_val (RMSE): ' + str(loss_val_final) + 'r2_val: ' + str(r2_val_final) + '\n')
file_handle.write('loss_test (RMSE): ' + str(loss_test_final) + 'r2_test: ' + str(r2_test_final) + '\n')
file_handle.write('Best Results:' + '\n')
file_handle.write('loss_train (RMSE): ' + str(loss_train_total[best_idx]) + 'r2_train: ' + str(r2_train_total[best_idx]) + '\n')
file_handle.write('loss_val (RMSE): ' + str(loss_val_total[best_idx]) + 'r2_val: ' + str(r2_val_total[best_idx]) + '\n')
file_handle.write('loss_test (RMSE): ' + str(loss_test_total[best_idx]) + 'r2_test: ' + str(r2_test_total[best_idx]) + '\n')
file_handle.close()
