import numpy as np
from sklearn.cluster import KMeans
import time


def clustering(data, n_clusters, verbose=0):
    """
    
    :param data: [n_samples, n_features]
    :param n_clusters: int, how many clusters that user want
    :param verbose, by default 0
    :return: centers, [n_clusters, n_features]
    """
    data_shape = data.shape
    center_shape = (n_clusters, ) + data_shape[1:]

    t0 = time.time()
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
    t1 = time.time()
    if verbose>0:
        print("Clustering 10 classes in {:.2f} seconds...".format(t1-t0))

    centers = np.zeros(shape=center_shape)
    labels = kmeans.labels_
    t0 = time.time()
    batch_size = int(n_clusters / 10)
    for i in range(10):
        r1 = i * batch_size
        r2 = (i + 1) * batch_size
        index = np.argwhere(labels == i).squeeze()
        c = data[index]
        kmeans = KMeans(n_clusters=batch_size, random_state=0).fit(c)
        centers[r1:r2] = kmeans.cluster_centers_
    t1 = time.time()
    if verbose > 0:
        print("Clustering 10*{:d} classes in {:.2f} seconds...".format(batch_size, t1-t0))
    return centers
