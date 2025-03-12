import numpy as np
from netrd.distance import DeltaCon
from scipy.spatial import procrustes
from scipy.linalg import norm
import networkx as nx

# Traditional Network Distances

def distance_frobenius(A, B):
    """
    Compute "Frobenius" distance between 2 matrices
    """

    dist = norm(A - B, 'fro')

    return dist


def distance_procrustes(A, B):
    """
    Compute "Procrustes" distance between 2 matrices
    """

    mtx1, mtx2, dist = procrustes(A, B)

    return dist

def DeltaConDistance(A, B):
    """
    Compute Delta Connectivity distance between 2 adjancency matrices
    """

    assert type(A) == np.ndarray
    assert type(A) == type(B)

    G1, G2 = nx.from_numpy_array(A), nx.from_numpy_array(B)
    metric = DeltaCon()
    d1 = metric.dist(G1=G1, G2=G2)
    return (d1)
