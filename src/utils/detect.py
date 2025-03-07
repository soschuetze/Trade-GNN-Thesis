import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
import torch
import argparse
from datetime import datetime
from pathlib import PosixPath
import os
from functions import load_model, load_sequence, prepare_batches
from misc import convert_labels_into_changepoints, NpEncoder, get_device
from metrics import adjust_predicts_donut, binary_metrics_adj, compute_ari
import ruptures as rpt
from model import SiameseGNN

def compute_sgnn_similarity(model, sequence, window_length):
    """
    :param model:
    :param sequence:
    :param window_length:
    :return:
    """

    batches = prepare_batches(sequence, window_length=window_length)
    avg_sim = [] # average similarity metric
    idx = [] # indices of times at which the average similarity is computed

    with torch.no_grad():
        for (graph1_batch, graph2_batch, i) in batches: # each batch contains a L similarity score within a window
            pred = 0
            for graph1, graph2 in zip(graph1_batch, graph2_batch):

                pred +=  model(graph1, graph2).item()

            avg_sim.append(pred/len(graph1_batch))
            idx.append(i[1].item())

    return np.array(avg_sim), np.array(idx)

  
def detect_change_point(args=None):
    # Load s-GNN model
    model = load_model(args.model_type, args.model_path)
    # Create subdirectory for results
    save_dir = (f'NCPD_{datetime.utcnow().strftime("%m_%d_%H:%M:%S")}'
                f'_window_{args.window_length}'
                f'_{args.task}'
                )
    save_dir = PosixPath(args.save_dir).expanduser() / save_dir
    
    # Load graph sequence
    data, true_labels, true_cps = load_sequence(args.test_data)
    if true_labels is not None:
        true_labels = convert_labels_into_changepoints(true_labels, tolerance=args.tolerance)
    avg_sim, idx = compute_sgnn_similarity(model, data, window_length=args.window_length)
    T = len(data)
    
    # Method 1: detect change-points when similarity below threshold
    est_labels = (avg_sim < args.threshold).astype(int)
    # add 0 labels for the first L time stamps
    est_labels = np.concatenate([np.zeros(args.window_length).astype(int), est_labels], axis=0)

    # Only select positive labels for which there is not a positive labels in the previous L time stamps
    no_cps_before = np.array([np.sum(est_labels[i:i+args.window_length]) for i in range(len(est_labels)-args.window_length)])
    est_labels = ((est_labels[args.window_length:] == 1) * (no_cps_before == 0)).astype(int)
    est_labels = np.concatenate([np.zeros(args.window_length).astype(int), est_labels], axis=0)
    est_cps = np.where(est_labels == 1)[0]
    print("Method 1: change points detected at times ", est_cps)

    # save similarity, change-point labels, and hyperparameters
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(save_dir / 'avg_similarity.p', 'wb') as f:
        pickle.dump(avg_sim, f)
    with open(save_dir / 'est_labels.p', 'wb') as f:
        pickle.dump(est_labels, f)
    with open(save_dir / 'est_cps.p', 'wb') as f:
        pickle.dump(est_cps, f)
    with open(save_dir / 'args.json', 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)
    with open(save_dir / 'times.p', 'wb') as f:
        pickle.dump(idx, f)

    # Numerical evaluation
    if true_labels is not None and args.task == 'detection':
        metrics_best = binary_metrics_adj(
            score=est_labels,
            target=np.array(true_labels),
            threshold=0.5,
            adjust_predicts_fun=adjust_predicts_donut,
            only_f1=False,
        )
        print(f"Test F1 score", metrics_best["f1"])
        ari = compute_ari(est_cps, true_cps, T)
        print(f"Test ARI: ", ari)
        if args.single:
            cp_error = args.window_length + 1 + np.argmin(avg_sim) - true_cps[0]
            print("Change point localisation error on a single change-point: ", cp_error)
            with open(save_dir / 'cp_error.json', 'wb') as f:
                pickle.dump(cp_error, f)
        with open(save_dir / 'classification_results.json', 'w') as f:
            json.dump(metrics_best, f, cls=NpEncoder)
        with open(save_dir / 'test_ari.json', 'w') as f:
            json.dump(ari, f, cls=NpEncoder)
    return str(save_dir), metrics_best, ari

    
def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data', type=str, default='../../results/explain_graphs', help='Path to dynamic network data.')
    parser.add_argument('--model_type', type=str, default='sage', help='Type of encoder to use with Siamese GNN')
    parser.add_argument('--single', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, default='../../models/real_data/mis_logged_with_gdp.pt', help='Path to model for detecting change-points.')
    parser.add_argument('--save_dir', type=str, default='../../results/explain_graphs/results', help='Name of folder where to store results')
    parser.add_argument('--window_length', type=int, default=2, help='Length of backward window')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold on the similarity statistic to detect change-points')
    parser.add_argument('--cuda', type=int, default=None, choices=[0, 1, 2, 3], help='GPU id')
    parser.add_argument('--tolerance', type=int, default=3, help='Tolerance level in the adjusted F1 metric')
    parser.add_argument('--task', type=str, default='detection', choices=['detection', 'statistic'])

    args = parser.parse_args()
    
  
    return args

def main():
    args = get_args()

    detect_change_point(args=args)


if __name__ == '__main__':
    main()