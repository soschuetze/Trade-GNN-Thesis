import numpy as np
import torch
from typing import Union
import json
from torch.utils.data import DataLoader
import pickle
import os
from utils.misc import collate
import torch
from model import SiameseGNN, GraphSAGE, GAT, GIN, GCN


def dist_labels_to_changepoint_labels(labels: Union[np.ndarray, list]):
    """
    Convert graph distribution labels (phase) to change-point labels (0 or 1)
    :param labels (list or np.ndarray):
    :return: np.ndarray
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    cps = np.concatenate([np.zeros(1).astype(int), (abs(labels[1:] - labels[:-1]) > 0).astype(int)],axis=0)
    return cps

def prepare_batches(data, window_length):

    tuples = []

    for i in range(window_length, len(data)):
        for j in range(1, window_length+1):
            tuples.append((data[i], data[i-j], i))
    batched_data = DataLoader(tuples, batch_size=window_length, shuffle=False, collate_fn=collate,
                               drop_last=False)

    return  batched_data

def normalise_statistics(statistics):
    """
    Transform a statistic in [a,b] into [0,1] by substracting min and dividing by max
    :param statistics (list or nd.array):
    :return:
    """
    norm_stat = np.array(statistics)
    norm_stat = norm_stat - np.min(norm_stat)
    norm_stat = norm_stat / np.max(norm_stat)
    return norm_stat

def load_sequence(datapath):
  
    if os.path.isfile(datapath):
        with open(datapath, 'rb') as f:
            data = pickle.load(f)
        time = None
        labels = None
    else:
        with open(datapath + '/sage-mis-norm-gdp-data.p', 'rb') as f:
            data = pickle.load(f)

        with open(datapath + '/sage-mis-norm-gdp-labels.p', 'rb') as f:
            labels = pickle.load(f)

        with open(datapath + '/cp-times.json') as f:
            time = json.load(f)

    print(f"Data loaded: sequence of {len(data)} graphs with a change point at time {time}")
  
    return data, labels, time

def load_model(model_path: str):

    encoder = GAT(27, 16, 0.1)
    encoder.load_state_dict(torch.load(f'{model_path}-encoder.pt'))
    model = SiameseGNN(encoder, 50, 27, dropout = 0.05, nhidden=16)
    model.load_state_dict(torch.load(f'{model_path}.pt'))

    return model
