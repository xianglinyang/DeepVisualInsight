from sklearn.neighbors import KDTree
import backend
import numpy as np
from scipy.stats import spearmanr
from pynndescent import NNDescent
from sklearn.manifold import trustworthiness


def evaluate_proj_nn_perseverance_knn(data, embedding, n_neighbors, metric="euclidean"):
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    # get nearest neighbors
    nnd = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    high_ind, _ = nnd.neighbor_graph
    nnd = NNDescent(
        embedding,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    low_ind, _ = nnd.neighbor_graph

    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i],low_ind[i]))

    return border_pres.mean(), border_pres.max(), border_pres.min()


def evaluate_proj_nn_perseverance_trustworthiness(data, embedding, n_neighbors, metric="euclidean"):
    t = trustworthiness(data, embedding, n_neighbors=n_neighbors, metric=metric)
    return t


def evaluate_proj_boundary_perseverance_knn(data, embedding, high_centers, low_centers, n_neighbors):
    high_tree = KDTree(high_centers)
    low_tree = KDTree(low_centers)

    _, high_ind = high_tree.query(data, k=n_neighbors)
    _, low_ind = low_tree.query(embedding, k=n_neighbors)
    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i], low_ind[i]))

    return border_pres.mean(), border_pres.max(), border_pres.min()


def evaluate_proj_temporal_perseverance(prev_data, prev_embedding, data, embedding, n_neighbors):
    alpha = backend.find_alpha(prev_data, data, n_neighbors)
    delta_x = np.linalg.norm(prev_embedding - embedding, axis=1)
    correlation, pvalue = spearmanr(alpha, delta_x)
    return correlation


def evaluate_inv_distance(data, inv_data):
    return np.linalg.norm(data-inv_data, axis=1).mean()


def evaluate_inv_accu():
    pass


def evaluate_inv_conf():
    pass
