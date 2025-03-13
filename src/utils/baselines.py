import numpy as np
from utils.distances import DeltaConDistance, distance_frobenius, distance_procrustes
from utils.metrics import adjust_predicts_donut, binary_metrics_adj, compute_ari
from utils.functions import dist_labels_to_changepoint_labels, normalise_statistics
import networkx as nx

def avg_deltacon_similarity(nx_graphs, window_length, diff=False):
    if not isinstance(nx_graphs[0], nx.Graph):
        raise ValueError("Data should be a list of NetworkX graphs.")
    data = [nx.to_numpy_array(g) for g in nx_graphs]

    avg_sim = []
    for i in range(window_length, len(data)):
        sim_t = []
        for j in range(1, window_length + 1):
            sim_t.append(DeltaConDistance(data[i], data[i - j]))
        avg_sim.append(np.mean(sim_t))

    if diff:
        d_avg_sim = np.abs(np.array(avg_sim[1:]) - np.array(avg_sim[:-1]))
        return d_avg_sim, np.arange(window_length+1, len(data))

    return np.array(avg_sim), np.arange(window_length, len(data))


def avg_frobenius_distance(nx_graphs, window_length, diff=False):
    if not isinstance(nx_graphs[0], nx.Graph):
        raise ValueError("Data should be a list of NetworkX graphs.")
    data = [nx.to_numpy_array(g) for g in nx_graphs]

    avg_dist = []
    for i in range(window_length, len(data)):
        dist_t = []
        for j in range(1, window_length + 1):
            dist_t.append(distance_frobenius(data[i], data[i - j]))
        avg_dist.append(np.mean(dist_t))

    if diff:
        d_avg_dist = np.abs(np.array(avg_dist[1:]) - np.array(avg_dist[:-1]))
        return d_avg_dist, np.arange(window_length+1, len(data))

    return np.array(avg_dist), np.arange(window_length, len(data))


def avg_procrustes_distance(nx_graphs, window_length, diff=False):
    if not isinstance(nx_graphs[0], nx.Graph):
        raise ValueError("Data should be a list of NetworkX graphs.")
    data = [nx.to_numpy_array(g) for g in nx_graphs]

    avg_dist = []
    for i in range(window_length, len(data)):
        dist_t = []
        for j in range(1, window_length + 1):
            dist_t.append(distance_procrustes(data[i], data[i - j]))
        avg_dist.append(np.mean(dist_t))

    if diff:
        d_avg_dist = np.abs(np.array(avg_dist[1:]) - np.array(avg_dist[:-1]))
        return d_avg_dist, np.arange(window_length + 1, len(data))

    return np.array(avg_dist), np.arange(window_length, len(data))


def evaluate_baseline(model, test_data, test_labels, window_length, metric='adjusted_f1', diff=False):
    """
    Evaluate SC-NCPD method on a dynamic network sequence, for a given metric and using a detection threshold selected on training sequence

    :param training_data: dynamic network sequence for selecting threshold
    :param training_labels: distribution labels of the training sequence
    :param test_data: dynamic network sequence for evaluation
    :param test_labels:
    :param window_length:
    :param metric:
    :return:
    """

    T_test = len(test_data)

    # Computes CP statistic on train and test sequence
    if model == 'deltacon':
        stat_test, stat_test_times = avg_deltacon_similarity(test_data, window_length=window_length, diff=diff)
    elif model == 'frobenius':
        stat_test, stat_test_times = avg_frobenius_distance(test_data, window_length=window_length, diff=diff)
    elif model == 'procrustes':
        stat_test, stat_test_times = avg_procrustes_distance(test_data, window_length=window_length, diff=diff)
    else:
        raise ValueError('Method not yet implemented')

    # Normalise the statistics in [0,1]
    stat_test_norm = normalise_statistics(stat_test)

    if model == 'deltacon': # compute 1 - stat for DeltaCon similarity
        stat_test_norm = 1. - stat_test_norm

    # Convert and adjust the distribution labels of the snaphots with the given tolerance level
    cp_lab_test = dist_labels_to_changepoint_labels(test_labels)[stat_test_times]
    
    thresh = 0.5
    test_score = binary_metrics_adj(score=stat_test_norm, target=cp_lab_test, threshold=thresh,
                                  adjust_predicts_fun=adjust_predicts_donut,
                                  only_f1=True) # adjusted f1 score
    # ARI
    det_cps = stat_test_times[np.where(stat_test_norm > thresh)[0]]
    cp_lab_test = dist_labels_to_changepoint_labels(test_labels)[stat_test_times]
    true_cps = stat_test_times[np.where(cp_lab_test == 1)[0]]
    test_ari = compute_ari(det_cps, true_cps, T_test)

    return test_ari, test_score, det_cps
