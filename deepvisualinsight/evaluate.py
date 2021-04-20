from sklearn.neighbors import KDTree
from deepvisualinsight import backend
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import pearsonr
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

    # return border_pres.mean(), border_pres.max(), border_pres.min()
    return border_pres.mean()


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

    # return border_pres.mean(), border_pres.max(), border_pres.min()
    return border_pres.mean()


def evaluate_proj_temporal_perseverance_corr(alpha, delta_x):
    alpha = alpha.T
    delta_x = delta_x.T
    shape = alpha.shape
    data_num = shape[0]
    corr = np.zeros(data_num)
    for i in range(data_num):
        # correlation, pvalue = spearmanr(alpha[:, i], delta_x[:, i])
        correlation, pvalue = pearsonr(alpha[i], delta_x[i])
        if np.isnan(correlation):
            correlation = 0.0
        corr[i] = correlation
    return corr.mean()


def evaluate_inv_distance(data, inv_data):
    return np.linalg.norm(data-inv_data, axis=1).mean()


def evaluate_inv_accu(labels, pred):
    return np.sum(labels == pred) / len(labels)


def evaluate_inv_conf(labels, ori_pred, new_pred):
    old_conf = [ori_pred[i, labels[i]] for i in range(len(labels))]
    new_conf = [new_pred[i, labels[i]] for i in range(len(labels))]
    old_conf = np.array(old_conf)
    new_conf = np.array(new_conf)

    diff = np.abs(old_conf - new_conf)
    # return diff.mean(), diff.max(), diff.min()
    return diff.mean()


def evaluate_inv_nn(data, recon, n_neighbors=15):
    n_trees = 5 + int(round((data.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(data.shape[0]))))
    # get nearest neighbors
    nnd = NNDescent(
        data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    high_ind, _ = nnd.neighbor_graph
    nnd = NNDescent(
        recon,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    recon_ind, _ = nnd.neighbor_graph

    border_pres = np.zeros(len(data))
    for i in range(len(data)):
        border_pres[i] = len(np.intersect1d(high_ind[i], recon_ind[i]))
    # return border_pres.mean(), border_pres.max(), border_pres.min()
    return border_pres.mean()


def evaluate_proj_temporal_perseverance_entropy(alpha, delta_x):
    alpha = alpha.T
    delta_x = delta_x.T
    shape = alpha.shape
    data_num = shape[0]
    # normalize
    # delta_x_norm = delta_x.max(-1)
    # delta_x_norm = (delta_x.T/delta_x_norm).T
    delta_x_norm = delta_x.max()
    delta_x_norm = delta_x / delta_x_norm

    alpha = np.floor(alpha*10)
    delta_x_norm = np.floor(delta_x_norm*10)

    corr = np.zeros(len(alpha))
    # samples
    for i in range(len(alpha)):
        # alpha0-alpha9
        index = list()
        entropy = list()
        for j in range(11):
            dx = delta_x_norm[i][np.where(alpha[i] == j)]
            entropy_x = np.zeros(11)
            for k in range(11):
                entropy_x[k] = np.sum(dx == k)
            if np.sum(entropy_x) == 0:
                continue
            else:
                entropy_x = entropy_x / np.sum(entropy_x+10e-8)
                entropy_x = np.sum(entropy_x*np.log(entropy_x+10e-8))
                entropy.append(-entropy_x)
                index.append(j)
        if len(index) < 2:
            print("no enough data to form a correlation, setting correlation to be 0")
            corr[i] = 0
        else:
            correlation, _ = pearsonr(index, entropy)
            corr[i] = correlation

    return corr.mean()
