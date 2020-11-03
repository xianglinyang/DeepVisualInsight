from sklearn.neighbors import KNeighborsClassifier
from deepview.embeddings import init_inv_umap
import scipy.spatial.distance as distan
import numpy as np
import umap

def leave_one_out_knn_dist_err(dists, labs, n_neighbors=5):
    nn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    nn.fit(dists, labs) 
    
    unique_l = np.unique(labs)
    errs = 0
    # calculate the leave one out nearest neighbour error for each point
    neighs = nn.kneighbors(return_distance=False)
    neigh_labs = labs[neighs]
    counts_cl = np.zeros([labs.shape[0], unique_l.shape[0]])
    for i in range(unique_l.shape[0]):
        counts_cl[:,i] = np.sum(neigh_labs  == unique_l[i], 1)
    
    pred_labs = unique_l[counts_cl.argmax(1)]
    
    # calculate the prediction error
    return sum(pred_labs != labs)/labs.shape[0]

def evaluate_umap(deepview, X, Y, return_values=False):
    if len(np.shape(X)) > 2:
        bs = len(X)
        X = X.reshape(bs, -1)

    neighbors = 30
    embedding_sup = deepview.embedded
    labs = deepview.y_true
    pred_labs = deepview.y_pred
    dists = deepview.distances

    umap_unsup = umap.UMAP(n_neighbors=neighbors, random_state=11*12*13)
    embedding_unsup = umap_unsup.fit_transform(X)

    eucl_dists = distan.pdist(X)
    eucl_dists = distan.squareform(eucl_dists)

    # calc dists in fish umap proj
    fishUmap_dists = distan.pdist(embedding_sup)
    fishUmap_dists = distan.squareform(fishUmap_dists)

    # calc dists in euclidean umap proj
    euclUmap_dists = distan.pdist(embedding_unsup)
    euclUmap_dists = distan.squareform(euclUmap_dists)

    label_eucl_err   = leave_one_out_knn_dist_err(eucl_dists, Y, n_neighbors=5)
    label_fish_err   = leave_one_out_knn_dist_err(dists, Y, n_neighbors=5)
    label_fishUm_err = leave_one_out_knn_dist_err(fishUmap_dists, Y, n_neighbors=5)
    label_euclUm_err = leave_one_out_knn_dist_err(euclUmap_dists, Y, n_neighbors=5)

    # comparison to classifier labels
    pred_eucl_err   = leave_one_out_knn_dist_err(eucl_dists, pred_labs, n_neighbors=5)
    pred_fish_err   = leave_one_out_knn_dist_err(dists, pred_labs, n_neighbors=5)
    pred_fishUm_err = leave_one_out_knn_dist_err(fishUmap_dists, pred_labs, n_neighbors=5)
    pred_euclUm_err = leave_one_out_knn_dist_err(euclUmap_dists, pred_labs, n_neighbors=5)

    if return_values:
        return {
            'true'  : { 
                'eucl'      : label_eucl_err,
                'fish'      : label_fish_err,
                'eucl_umap' : label_euclUm_err,
                'fish_umap' : label_fishUm_err },
            'pred'  : { 
                'eucl'      : pred_eucl_err,
                'fish'      : pred_fish_err,
                'eucl_umap' : pred_euclUm_err,
                'fish_umap' : pred_fishUm_err }
        }
    else:
        print("orig labs, knn err: eucl / fish", label_eucl_err, "/", label_fish_err)
        #print("eucl / fish / fish umap proj knn err", label_eucl_err, "/", label_fish_err, "/", label_fishUm_err)
        print("orig labs, knn err in proj space: eucl / fish", label_euclUm_err, "/", label_fishUm_err)
        print("classif labs, knn err: eucl / fish", pred_eucl_err, "/", pred_fish_err)
        print("classif labs, knn acc in proj space: eucl / fish", 
            '%.1f'%(100 -100*pred_euclUm_err), "/", 
            '%.1f'%(100 -100*pred_fishUm_err))


def evaluate_inv_umap(deepview, X, Y, train_frac=.7):
    n_samples = len(X)
    n_train = int(n_samples * train_frac)

    deepview.reset()
    deepview.max_samples = n_samples
    deepview.add_samples(X, Y)

    # pick samples for training and testing
    train_samples = deepview.samples[:n_train]
    train_embeded = deepview.embedded[:n_train]
    train_labels = deepview.y_pred[:n_train]
    test_samples = deepview.samples[n_train:]
    test_embeded = deepview.embedded[n_train:]
    test_labels = deepview.y_pred[n_train:]

    # get DeepView an untrained inverse mapper 
    # and train it on the train set
    deepview.inverse = init_inv_umap()
    deepview.inverse.fit(train_embeded, train_samples)

    # apply inverse mapping to embedded samples and
    # predict the reconstructions
    train_recon = deepview.inverse(train_embeded)
    train_preds = deepview.model(train_recon).argmax(-1)

    # calculate train accuracy
    n_correct = np.sum(train_labels == train_preds)
    train_acc = 100 * n_correct / n_train

    # evaluate on test set
    test_recon = deepview.inverse(test_embeded)
    test_preds = deepview.model(test_recon).argmax(-1)

    # calculate test accuracy
    n_correct = np.sum(test_labels == test_preds)
    test_acc = 100 * n_correct / len(test_labels)

    return train_acc, test_acc