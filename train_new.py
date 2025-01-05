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
from models import GCN, C_GCN

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
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=2000,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
threshold = 0.2
num_edge = None
seed = 42
idx_anchors = range(25)
idx_agents = range(25, 500)

# Refined Functions
def get_Gamma(alpha, beta, S):
    device = S.device
    return beta * torch.eye(S.size(0), device=device) - beta * alpha * S

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

def lp_refine(idx_test, idx_train, labels, output, S, alpha=1., beta=1.):
    Gamma = get_Gamma(alpha, beta, S)

    pred_train = output[idx_train]
    pred_test = output[idx_test]
    res_pred_train = labels[idx_train] - output[idx_train]
    
    refined_test = pred_test + interpolate(idx_train, idx_test, res_pred_train, Gamma)

    return refined_test

# Training Variable Initialization
coeffs = Variable(torch.FloatTensor([1., 3.0]).cuda() if args.cuda else torch.FloatTensor([1., 3.0]) , requires_grad=True)

# Fix the random seed
np.random.seed(seed)
torch.manual_seed(seed)
if args.cuda:
    torch.cuda.manual_seed(seed)
    
# Load data
data_path = "./Networks_new/Uniform_100samples_25anchors_475agents_0Percent_0Variance.mat"
mode_fea, mode_adj, idx_train, idx_val, idx_test, features_all, labels_all, adj_all, S_all, fea_true_all, truncated_noise_all, range_anchors, range_agents = load_data(data_path, threshold, num_edge)

# Model
model = GCN(nfeat=features_all[0].shape[1],
                nhid=[2000],
                nout=labels_all[0].shape[1],
                dropout=args.dropout)
# model = C_GCN(nfeat=features_all[0].shape[1],
#                 nhid1=args.hidden,
#                 nout=labels_all[0].shape[1],
#                 dropout=args.dropout)

# Optimizer
coeffs_optimizer = optim.SGD([coeffs], lr=1e-3, momentum=0.0)
optimizer = optim.Adam(model.parameters(),
                        lr=args.lr, weight_decay=args.weight_decay)

# Criterion
# Training loss function
def setdiff(n, idx):
    idx = idx.cpu().detach().numpy()
    cp_idx = np.setdiff1d(np.arange(n), idx)
    return cp_idx

def loss_corr(output, labels, idx, S, coeffs, add_logdet):
    output = output.reshape(-1, 1)
    labels = labels.reshape(-1, 1)
    rL = labels[idx] - output[idx]
    S = S.to_dense()
    I = torch.eye(S.size(0)).cuda()
    
    Gamma = (I - torch.tanh(coeffs[0])*S)*torch.exp(coeffs[1])
    cp_idx = setdiff(len(S), idx)
    loss1 = torch.matmul(rL.T,torch.matmul(Gamma[idx, :][:, idx], rL) - torch.matmul(Gamma[idx, :][:, cp_idx], torch.matmul(torch.inverse(Gamma[cp_idx, :][:, cp_idx]), torch.matmul(Gamma[cp_idx, :][:, idx], rL))))
    loss2 = torch.Tensor([0.]).cuda() if args.cuda else torch.Tensor([0.])
    if add_logdet: loss2 = logdet(Gamma) - logdet(Gamma[cp_idx, :][:, cp_idx])
    l = loss1 - loss2

    return l/len(idx)
# RMSE loss function
loss_fun = torch.nn.MSELoss()
# Diatance
# def compute_distance_matrix(points):
#     diff = points[:, None, :] - points[None, :, :]
#     dist_matrix = torch.sqrt(torch.sum(diff ** 2, dim=-1))
#     return dist_matrix
def compute_distance_matrix(A, B):
    A_expanded = A.unsqueeze(1)
    B_expanded = B.unsqueeze(0)

    dist_sq = torch.sum((A_expanded - B_expanded) ** 2, dim=2)
    dist = torch.sqrt(dist_sq)
    return dist
# R2 Function
def R2(outputs, labels):
    outputs = outputs.cpu().detach().numpy().reshape(-1)
    labels = labels.cpu().detach().numpy().reshape(-1)
    return r2_score(labels, outputs)

# Transfer data to cuda
if args.cuda:
    model.cuda()
    range_anchors = range_anchors.cuda()
    range_agents = range_agents.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    idx_anchors = torch.LongTensor(idx_anchors).cuda()
    idx_agents = torch.LongTensor(idx_agents).cuda()
    labels_all = labels_all.cuda()
    for k in range(100):
        features_all[k] = features_all[k].cuda()
        adj_all[k] = adj_all[k].cuda()
        S_all[k] = S_all[k].cuda()
        fea_true_all[k] = fea_true_all[k].cuda()
    
# Train
def train(features, adj, labels, features_true, sign):
    t = time.time()
    model.train()
    # model.gc1.weight.requires_grad = True
    # model.gc1.bias.requires_grad = True
    # model.gc2.weight.requires_grad = True
    # model.gc2.bias.requires_grad = True
    # model.coeff.requires_grad = False
    optimizer.zero_grad()
    output = model(features, adj)
    # output = model(features, adj, labels, idx_anchors, idx_agents)
    # output = model(features, adj, labels, idx_anchors, idx_agents, coeffs[0], coeffs[1])
    
    if sign == 0:
        dist_anchors = compute_distance_matrix(output[idx_anchors], labels[idx_anchors])
        output_cat = torch.cat((output[idx_anchors], 0.5 * dist_anchors), dim=1)
        labels_cat = torch.cat((labels[idx_anchors], 0.5 * range_anchors), dim=1)
        loss_train = loss_fun(output_cat, labels_cat) # 加上feature矩阵
    elif sign == 1:
        loss_train = loss_fun(output[idx_anchors], labels[idx_anchors])
    
    # dist = compute_distance_matrix(output, labels[idx_anchors])
    # loss_train = loss_fun(output[idx_anchors], labels[idx_anchors]) + loss_fun(dist, range_agents)
    
    # loss_train = loss_fun(output[idx_anchors], labels[idx_anchors])
    # loss_train = loss_corr(output, labels, idx_anchors, adj, coeffs, False)
    loss_train.backward()
    optimizer.step()

    loss_anchors = torch.sqrt(loss_fun(output[idx_anchors], labels[idx_anchors]))
    loss = torch.sqrt(loss_fun(output[idx_agents], labels[idx_agents]))
    r2 = R2(output[idx_agents], labels[idx_agents])
    
    # if epoch % 10 == 0:
    #     model.train()
    #     # model.gc1.weight.requires_grad = False
    #     # model.gc1.bias.requires_grad = False
    #     # model.gc2.weight.requires_grad = False
    #     # model.gc2.bias.requires_grad = False
    #     # model.coeff.requires_grad = True
    #     coeffs_optimizer.zero_grad()
    #     # optimizer.zero_grad()
    #     # output = model(features, adj, labels, idx_anchors, idx_agents)
    #     output = model(features, adj, labels, idx_anchors, idx_agents, coeffs[0], coeffs[1])
        
    #     # dist_anchors = compute_distance_matrix(output[idx_anchors], labels[idx_anchors])
    #     # output_cat = torch.cat((output[idx_anchors], dist_anchors), dim=1)
    #     # labels_cat = torch.cat((labels[idx_anchors], range_anchors), dim=1)
    #     # loss_train = loss_fun(output_cat, labels_cat)
        
    #     loss_train = loss_fun(output[idx_anchors], labels[idx_anchors])
    #     # loss_train = loss_corr(output, labels, idx_train, adj, coeffs, True)
    #     loss_train.backward()
    #     coeffs_optimizer.step()
    #     # optimizer.step()
    
    # Evaluate the model
    if not args.fastmode:
        model.eval()
        output = model(features, adj)
        # output = model(features, adj, labels, idx_anchors, idx_agents)
        # output = model(features, adj, labels, idx_anchors, idx_agents, coeffs[0], coeffs[1])
    
    loss_val = torch.sqrt(loss_fun(output[idx_agents], labels[idx_agents]))
    r2_val = R2(output[idx_agents], labels[idx_agents])
    
    return loss_anchors, loss, r2, loss_val, r2_val

# Test
def test(features, adj, labels, features_true):
    model.eval()
    output = model(features, adj)
    # output = model(features, adj, labels, idx_anchors, idx_agents)
    # output = model(features, adj, labels, idx_anchors, idx_agents, coeffs[0], coeffs[1])
    # Label Propagation
    # prediction_raw = lp_refine(idx_agents, idx_anchors, labels, output, adj)
    # Maximize Likelihood
    # prediction = lp_refine(idx_agents, idx_anchors, labels, output, adj, torch.tanh(coeffs[0]).item(), torch.exp(coeffs[1]).item())
    
    # loss = torch.sqrt(loss_fun(prediction, labels[idx_agents]))
    # loss = torch.sqrt(loss_fun(prediction_raw, labels[idx_agents]))
    loss = torch.sqrt(loss_fun(output[idx_agents], labels[idx_agents]))
    # loss = loss(output, labels, idx_agents, adj, coeffs, True)
    r2 = R2(output[idx_agents], labels[idx_agents])
    # r2_test_refine = R2(prediction, labels[idx_agents])
    # r2_test_refine_raw = R2(prediction_raw, labels[idx_agents])
    return loss, r2

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
sign = 1

print("\nTraining!\n")
for epoch in range(args.epochs):
    if epoch >= 100:
        sign = 0
    else:
        sign = 1
    # Train the model
    t_epoch = time.time()
    loss_anchors = []
    loss_train = []
    r2_train = []
    if not args.fastmode:
        loss_val = []
        r2_val = []
    for batch in idx_train:
        loss_anchors_tem, loss_train_tem, r2_train_tem, loss_val_tem, r2_val_tem = train(features_all[batch], adj_all[batch], labels_all[batch], fea_true_all[batch], sign)
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
        loss_test_tem, r2_test_tem = test(features_all[batch], adj_all[batch], labels_all[batch], fea_true_all[batch])
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
            # "alpha= {:.4f}".format(torch.tanh(coeffs[0]).item()),
            # "beta= {:.4f}".format(torch.exp(coeffs[1]).item())
            'time: {:.4f}s'.format(time.time() - t_epoch)
            )

loss_anchors_final = sum(loss_anchors_total) / args.epochs
loss_train_final = sum(loss_train_total) / args.epochs
r2_train_final = sum(r2_train_total) / args.epochs
loss_val_final = sum(loss_val_total) / args.epochs
r2_val_final = sum(r2_val_total) / args.epochs
loss_test_final = sum(loss_test_total) / args.epochs
r2_test_final = sum(r2_test_total) / args.epochs
# alpha_final = sum(alpha) / args.epochs
# beta_final = sum(beta) / args.epochs

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
    #   "alpha= {:.4f}".format(alpha_final), 
    #   "beta= {:.4f}".format(beta_final)
    )