import argparse
import numpy as np
import networkx as nx
import torch
import pickle
import dgl
import os
from tqdm import trange
from os.path import join
import random
from typing import Union
import os
import torch
import networkx as nx
from torch_geometric.data import Data
import scipy.sparse as ss
import scipy

def degree_matrix(A, sparse=True):
    """Returns the absolute degree matrix of a signed graph
    Args: A (csc or dense matrix): signed adjacency matrix
          sparse (boolean): sparse or dense matrix input and output"""

    if not sparse:
        return np.diag(abs(A).sum(axis=0), 0)

    else:
        return ss.diags(np.array(abs(A).sum(axis=0)).squeeze(), offsets=0).tocsc()

def identity(n, sparse=True):
    """
    Returns identity matrix of size n in sparse (ss.csr) or non-sparse (np.ndarray) format

    :param n (int):
    :param sparse (bool):
    :return:
    """

    if not sparse:
        return np.eye(n)
    else:
        return ss.eye(n)


def laplacian(A, sparse=True):
    """Returns the Laplacian matrix of a graph
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output"""

    return degree_matrix(A, sparse) - A


def norm_laplacian(A, sparse=True):
    """Returns the symmetric normalized Laplacian matrix
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output"""

    D = degree_matrix(np.abs(A), sparse)

    if not sparse:
        diag = np.diag(D)
        if diag.any() == 0.:
            print("Regularising graph with isolated nodes")
            diag = diag + 0.01 * np.ones(A.shape[0])
        Dinv = np.diag( 1.0 / np.sqrt(diag))
        #Dinv = np.linalg.inv(np.sqrt(D))
        return identity(A.shape[0]) - (Dinv.dot( A ).dot( Dinv ))

    else:
        diag = D.diagonal()
        if diag.any() == 0.:
            print("Regularising graph with isolated nodes")
            diag = diag + 0.01 * np.ones(A.shape[0])
        Dinv = ss.diags(1.0 / np.sqrt(diag))
        #Dinv = D.power(-0.5)
        return identity(A.shape[0]) - (Dinv @ A @ Dinv)




def laplacian_embeddings(A, k=None, sparse=True, normalize=True):
    """
    Returns the (sym) Laplacian embeddings matrix of dimension k. If the graph is signed, uses the signed Laplacian
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output
    output: N x k matrix
    """

    if normalize:
        L = norm_laplacian(A, sparse)
    else:
        L = laplacian(A, sparse)
    if k is None:
        k = A.shape[0] // 2
    if sparse:
        _, V = ss.linalg.eigsh(L, k=k, which='SA')
    else:
        _, V = scipy.linalg.eigh(L, subset_by_index=[0,k-1])

    return V

def random_walk_embeddings(A, k=None, sparse=True):
    """
    Returns the (sym) Laplacian embeddings matrix of dimension k. If the graph is signed, uses the signed Laplacian
    Args: A (csc or dense matrix): adjacency matrix
          sparse (boolean): sparse or dense matrix output
    output: N x k matrix
    """

    if k is None:
        k = 1
    P = np.zeros((A.shape[0], k))
    R = np.dot(degree_matrix(A, sparse), A)
    for i in range(1, k+1):
        Q = np.linalg.matrix_power(a=R.todense(), n=i)
        P[:,i-1] = np.diag(Q, k=0)

    return P

def which_community(node, sizes):
    """
    Returns the community of a node i in a networkx SBM graph generated using
    `nx.generators.community.stochastic_block_model(sizes, p, nodelist=None)`.
    """
    return np.where(node < np.cumsum(sizes))[0][0]



def nxSBM(num_nodes, type, sizes, p, q, features=None, signal_means=None, signal_std=1.0):
    """Generates attributed SBM graphs with Networkx library.

    Creates a graph with sum(sizes) nodes. The nodes are ordered 0,1,...,n-1 and are also ordered by their community
    membership. I.e. the memberships will look like 0,0,...,0,0,1,1,....1,1,2,2,... etc.

    sizes: list of community sizes
    features: None or mixture of gaussian
    signal_means: list of vectors for each community. A vector represents the means of a multidimensional signal.
    p,q: intra/inter-cluster link probability
    signal_std: standard deviation of the signal
    """
    n = len(sizes)
    C = q * np.ones((n, n))
    np.fill_diagonal(C, p)
    graph = nx.generators.community.stochastic_block_model(sizes, C)

    if features == 'gaussian' and signal_means is not None:
        attributes = []

        for node in graph.nodes():

            community = which_community(node, sizes)
            loc = signal_means[community]
            if isinstance(loc, np.ndarray) or isinstance(loc, list):
                scale = signal_std * np.ones(len(loc))
            else:
                scale = signal_std * np.ones(1)
            signal = np.random.normal(loc=loc, scale=scale)
            graph.nodes[node]['node_attr'] = signal
            attributes.append(np.array([[graph.degree[node]]]))

        attributes = np.concatenate(attributes, axis=0)

        return graph, attributes

    return graph, None

def nxER(num_nodes, type, sizes, p, q, features=None, signal_means=None, signal_std=1.0):
    n = num_nodes

    graph = nx.erdos_renyi_graph(n, q)

    return graph



def sample_pygcn_graph(sbm_args):

    if sbm_args['type'] == 'er':
        nx_er_graph = nxER(**sbm_args)
        adjacency = nx.to_scipy_sparse_array(nx_er_graph, format='csr')
        edge_index = torch.tensor(adjacency.nonzero(), dtype=torch.long)
        graph = Data(edge_index=edge_index)

    else:
        nx_sbm_graph, _ = nxSBM(**sbm_args) # if generated as arrays
        adjacency = nx.to_scipy_sparse_array(nx_sbm_graph, format='csr')
        edge_index = torch.tensor(adjacency.nonzero(), dtype=torch.long)
        graph = Data(edge_index=edge_index)

    return graph


def sample_pairs(seq, labels, nsamples=np.Inf, filename=None):
    """
    Samples pairs of graphs with labels (0=dissimilar, 1=similar) from a sequence of graphs
    :param seq: list of torch geometric graphs
    :param labels: array of labels of each graph indicating their generative distribution
    :param nsamples: number of pairs to sample
    :param filename: if None do not save the pairs
    :return: list of triplets (G_1, G_2, label)
    """

    pairs = np.triu_indices(len(seq), k=1)

    if nsamples == np.Inf:
        nsamples = len(pairs[0])

    idx_pairs = np.random.choice(range(pairs[0].shape[0]), min(10*nsamples, len(pairs[0])), replace=False) # random sample of indices in pairs
    data = []
    # Control class balance
    npos = 0
    nneg = 0
    i = -1
    while npos + nneg < nsamples and i < idx_pairs.shape[0]-1:
            i += 1
            id1 = pairs[0][idx_pairs[i]]
            id2 = pairs[1][idx_pairs[i]]

            lab = torch.Tensor([(labels[id1] == labels[id2])])
            if lab < 0.05 and nneg < nsamples//2:
                nneg+=1
            elif lab > 0.05 and npos < nsamples//2:
                npos+=1
            else:
                pass
            rdm = random.random()
            if rdm < 0.5:
                data.append([seq[id1], seq[id2], lab])
            else:
                data.append([seq[id2], seq[id1], lab])
            #print(id1, id2, lab)

    print("{} positive and {} negative examples".format(npos, nneg))
    random.shuffle(data)

    # save data
    if filename is not None:
        os.makedirs('../data', exist_ok=True)
        with open(join('../data', f'{filename}.p'), 'wb') as f:
            pickle.dump(data, f)

    return data



def sample_pairs_in_window(sequence, labels, window_length=6, n_pairs=None, path=None):
    """
    Sample all or a subsample of pairs of graphs in a sequence using a sliding window

    """

    n_data = len(sequence)
    npos = 0
    nneg = 0

    pairs = []

    for i in range(window_length, n_data):
        for j in range(1, window_length + 1):

            lab = torch.Tensor([(labels[i] == labels[i-j]).astype(int)])

            if lab < 0.05:
                nneg += 1
                pairs.append((sequence[i], sequence[i-j], lab))
            elif lab > 0.05:
                npos += 1
                pairs.append((sequence[i], sequence[i-j], lab))
            else:
                pass

    print("{} positive and {} negative examples".format(npos, nneg))
    if n_pairs is not None:
        pairs = random.sample(pairs, n_pairs)

    random.shuffle(pairs)

    if path is not None:
        with open(path, 'wb') as f:
            pickle.dump(pairs, f)

    return pairs

