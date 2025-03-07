import pandas as pd
import numpy as np
import pickle as pkl
import torch
import datetime as datetime
from tqdm import tqdm
import argparse
import itertools
import networkx as nx
import random
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from model import GraphSAGE, GCN, GAT, GIN, SiameseGNN
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, fbeta_score

def load_data(args):
    feats = args.feature_type
    with open(f'../../sGNN_pickle_files/{feats}/train_data_{feats}.pkl', 'rb') as f:
        graph_pairs_train = pkl.load(f)

    with open(f'../../sGNN_pickle_files/{feats}/val_data_{feats}.pkl', 'rb') as f:
        graph_pairs_val = pkl.load(f)
        
    with open(f'../../sGNN_pickle_files/{feats}/test_data_{feats}.pkl', 'rb') as f:
        graph_pairs_test = pkl.load(f)

    return graph_pairs_train, graph_pairs_val, graph_pairs_test

def train_model(args, lr, training_data_pairs, val_data_pairs, num_epochs=10):
    
    val_f1 = 0
    val_f2 = 0
    val_f05 = 0

    val_loss_arr = []
    train_loss_arr = []
        
    feats = args.feature_type
    model_type = args.model_type
    lr = args.learning_rate
    dropout = args.dropout
    sort_k = args.sort_k
    hidden_units = args.hidden_units
    input_dim = training_data_pairs[0][0].x.shape[1]

    model = SiameseGNN(sort_k, input_dim, dropout, nhidden=hidden_units)
    optimizer = optim.Adam(SiameseGNN.parameters(), lr, weight_decay=0.0001)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.BCELoss() 

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_losses = []
        for (graph1, graph2, labels) in training_data_pairs:
            optimizer.zero_grad()
            out = model(graph1, graph2)

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_losses = []
            val_pred = []
            val_truth = []

            correct = 0
            total = 0
            for (graph1, graph2, labels) in val_data_pairs:
                out = model(graph1, graph2)

                val_loss = criterion(out, labels)
                val_losses.append(val_loss.item())

                predictions = torch.round(out)

                val_pred.extend(predictions.cpu().numpy())
                val_truth.extend(labels.cpu().numpy())

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

            val_loss = sum(val_losses) / len(val_losses)
            val_accuracy = correct / total

        val_f1 = f1_score(val_truth, val_pred)
        val_f2 = fbeta_score(y_true=val_truth, y_pred=val_pred, beta=2)
        val_f05 = fbeta_score(y_true=val_truth, y_pred=val_pred, beta=1 / 2)

        train_loss_arr.append(sum(train_losses) / len(train_losses))
        val_loss_arr.append(val_loss)
        print(f'Epoch: {epoch + 1}, Training Loss: {sum(train_losses) / len(train_losses)}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, F1 Score: {val_f1}, F2 Score: {val_f2}, F0.5 Score: {val_f05}')

    model_name = f"models/{type}-{feats}.pt"
    torch.save(model.state_dict(), model_name)

    return train_losses, val_losses

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_dir', type=str, default='../pygcn/graphs_nogdp.pkl', help='Name of folder with initial graphs')
    parser.add_argument('--save_dir', type=str, default='../../sGNN_pickle_files/nogdp_randnorm', help='Name of folder where to store results')
    parser.add_argument('--gdp', type=str, default='nogdp', help='Whether GDP is included')
    parser.add_argument('--model_type', type=str, default='sage', help='Whether GDP is included')
    parser.add_argument('--feature_type', type=str, default='rand_norm', help='Types of features to be added')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Types of features to be added')
    parser.add_argument('--dropout', type=float, default=0.1, help='Types of features to be added')
    parser.add_argument('--sort_k', type=int, default=50, help='Types of features to be added')
    parser.add_argument('--hidden_units', type=int, default=16, help='Types of features to be added')

    args = parser.parse_args()
    
    return args

def main():

    torch.manual_seed(42)
    args = get_args()

    train, val, test = load_data(args)
    train_model(args, train, val, test)

if __name__ == '__main__':
    main()