""" help functions to evaluate projection function and inverse function"""
__Author__ = "xianglin"


def eval_proj_trustworthiness(X, X_embedded, n_neighbors=5, metric="euclidean"):
    from sklearn.manifold import trustworthiness
    return trustworthiness(X, X_embedded, n_neighbors=n_neighbors, metric=metric)


def eval_proj_kmeans(X, X_embedded, n_neighbors=[.01, .03, .05, .1], metric="euclidean"):
    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    n = X.shape[0]
    test_num = [int(x * n) for x in n_neighbors]

    ans = list()

    for neighbors in test_num:
        tmp = list()
        high = NearestNeighbors(n_neighbors=neighbors, metric=metric)
        high.fit(X)
        high_dist = high.kneighbors(return_distance=False)

        low = NearestNeighbors(n_neighbors=neighbors, metric=metric)
        low.fit(X_embedded)
        low_dist = low.kneighbors(return_distance=False)

        for i in range(len(X)):
            tmp.append(len(np.intersect1d(high_dist[i], low_dist[i])) / float(neighbors))

        ans.append(float(sum(tmp)) / len(tmp))
        print("Finish calculating {}-th neighbors for {} data points...".format(neighbors, n))

    return ans


def eval_inverse():