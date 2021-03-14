import numpy as np
from warnings import warn, catch_warnings, filterwarnings
from umap.umap_ import make_epochs_per_sample
from numba import TypingError
import os
from umap.spectral import spectral_layout
from sklearn.utils import check_random_state
import codecs, pickle
from sklearn.neighbors import KDTree
import tensorflow as tf
from sklearn.cluster import KMeans


def get_graph_elements(graph_, n_epochs):
    """
    gets elements of graphs, weights, and number of epochs per edge

    Parameters
    ----------
    graph_ : scipy.sparse.csr.csr_matrix
        umap graph of probabilities
    n_epochs : int
        maximum number of epochs per edge

    Returns
    -------
    graph scipy.sparse.csr.csr_matrix
        umap graph
    epochs_per_sample np.array
        number of epochs to train each sample for
    head np.array
        edge head
    tail np.array
        edge tail
    weight np.array
        edge weight
    n_vertices int
        number of verticies in graph
    """
    ### should we remove redundancies () here??
    # graph_ = remove_redundant_edges(graph_)

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200
    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = make_epochs_per_sample(graph.data, n_epochs)

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


def make_balance_per_sample(edges_to_exp, edges_from_exp, cp_num, centers_num, bc_num):
    balance_per_sample = np.zeros(shape=(len(edges_to_exp)))
    balance_per_sample[(edges_to_exp < cp_num) & (edges_from_exp < cp_num)] = 1
    # balance_per_sample[(edges_to_exp < (cp_num+centers_num)) & (edges_from_exp >= (cp_num + centers_num))] = 1
    # balance_per_sample[(edges_to_exp >= (cp_num + centers_num)) & (edges_from_exp < (cp_num+centers_num))] = 1
    # balance_per_sample[(edges_to_exp < 5000) & (edges_from_exp >= 5000)] = 1
    # balance_per_sample[(edges_to_exp >= 5000) & (edges_from_exp < 5000)] = 1

    return balance_per_sample


def construct_edge_dataset(
    X_input, graph_, n_epochs, batch_size, parametric_embedding, parametric_reconstruction,
):
    """
    Construct a tf.data.Dataset of edges, sampled by edge weight.

    Parameters
    ----------
    X : list, [X, DBP_samples]
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        DBP_samples : array, shape(n_samples, n_features)
            distant border points to be transformed.
    graph_ : scipy.sparse.csr.csr_matrix
        Generated UMAP graph
    n_epochs : int
        # of epochs to train each edge
    batch_size : int
        batch size
    parametric_embedding : bool
        Whether the embedder is parametric or non-parametric
    parametric_reconstruction : bool
        Whether the decoder is parametric or non-parametric
    """

    def gather_X(edge_to, edge_from, weight):
        fitting_data = np.concatenate((X, dbp), axis=0)
        edge_to_batch = tf.gather(fitting_data, edge_to)
        edge_from_batch = tf.gather(fitting_data, edge_from)

        outputs = {"umap": 0}
        if parametric_reconstruction:
            # add reconstruction to iterator output
            # edge_out = tf.concat([edge_to_batch, edge_from_batch], axis=0)
            outputs["reconstruction"] = edge_to_batch

        return (edge_to_batch, edge_from_batch, weight), outputs

    X, dbp = X_input
    tp_num = len(X)
    dbp_num = len(dbp)

    # get data from graph
    graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
        graph_, n_epochs
    )

    # number of elements per batch for embedding
    if batch_size is None:
        # batch size can be larger if its just over embeddings
        if parametric_embedding:
            batch_size = np.min([n_vertices, 1000])
        else:
            batch_size = len(head)

    edges_to_exp, edges_from_exp = (
        np.repeat(head, epochs_per_sample.astype("int")),
        np.repeat(tail, epochs_per_sample.astype("int")),
    )

    weight = np.repeat(weight, epochs_per_sample.astype("int"))

    # tptp_edges_num = np.sum((edges_to_exp < tp_num) & (edges_from_exp < tp_num))
    # dbpdbp_edges_num = np.sum((edges_to_exp >= tp_num) & (edges_from_exp >= tp_num))
    # tpdbp_edges_num = np.sum((edges_to_exp < tp_num) & (edges_from_exp >= tp_num)) + np.sum((edges_to_exp >= tp_num) & (edges_from_exp < tp_num))


    # balance_per_sample = make_balance_per_sample(edges_to_exp, edges_from_exp, tp_num, tp_num, dbp_num)
    #
    # edges_to_exp, edges_from_exp = (
    #     np.repeat(edges_to_exp, balance_per_sample.astype("int")),
    #     np.repeat(edges_from_exp, balance_per_sample.astype("int")),
    # )
    # weight = np.repeat(weight, balance_per_sample.astype("int"))
    #

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
    weight = weight[shuffle_mask].astype(np.float64)
    weight = np.expand_dims(weight, axis=1)

    # create edge iterator
    edge_dataset = tf.data.Dataset.from_tensor_slices(
        (edges_to_exp, edges_from_exp, weight)
    )
    edge_dataset = edge_dataset.repeat()
    edge_dataset = edge_dataset.shuffle(10000)
    edge_dataset = edge_dataset.map(
        # gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
        gather_X
    )
    edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
    edge_dataset = edge_dataset.prefetch(10)

    return edge_dataset, batch_size, len(edges_to_exp), head, tail, weight


def construct_mixed_edge_dataset(
    X_input, old_graph_, new_graph_, n_epochs, batch_size, parametric_embedding,
        parametric_reconstruction
):

    def gather_X(edge_to, edge_from, weight):
        edge_to_batch = tf.gather(fitting_data, edge_to)
        edge_from_batch = tf.gather(fitting_data, edge_from)

        outputs = {"umap": 0}
        if parametric_reconstruction:
            # add reconstruction to iterator output
            # edge_out = tf.concat([edge_to_batch, edge_from_batch], axis=0)
            outputs["reconstruction"] = edge_to_batch

        return (edge_to_batch, edge_from_batch, weight), outputs

    train_data, centers, border_centers = X_input
    cp_num = len(train_data)
    centers_num = len(centers)
    bc_num = len(border_centers)
    fitting_data = np.concatenate((train_data, centers, border_centers), axis=0)

    # get data from graph
    old_graph, old_epochs_per_sample, old_head, old_tail, old_weight, old_n_vertices = get_graph_elements(
        old_graph_, n_epochs
    )
    # get data from graph
    new_graph, new_epochs_per_sample, new_head, new_tail, new_weight, new_n_vertices = get_graph_elements(
        new_graph_, n_epochs
    )
    ## normalize two graphs
    new_head = new_head + cp_num
    new_tail = new_tail + cp_num

    # number of elements per batch for embedding
    if batch_size is None:
        # batch size randomly choose a number as batch_size if batch_size is None
        batch_size = 1000
    edges_to_exp = np.concatenate((np.repeat(old_head, old_epochs_per_sample.astype("int")),
                                   np.repeat(new_head, new_epochs_per_sample.astype("int"))), axis=0)
    edges_from_exp = np.concatenate((np.repeat(old_tail, old_epochs_per_sample.astype("int")),
                                   np.repeat(new_tail, new_epochs_per_sample.astype("int"))), axis=0)

    weight = np.concatenate((np.repeat(old_weight, old_epochs_per_sample.astype("int")),
                             np.repeat(new_weight, new_epochs_per_sample.astype("int"))), axis=0)

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
    weight = weight[shuffle_mask].astype(np.float64)
    weight = np.expand_dims(weight, axis=1)

    # create edge iterator
    edge_dataset = tf.data.Dataset.from_tensor_slices(
        (edges_to_exp, edges_from_exp, weight)
    )
    edge_dataset = edge_dataset.repeat()
    edge_dataset = edge_dataset.shuffle(10000)
    edge_dataset = edge_dataset.map(
        # gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
        gather_X
    )
    edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
    edge_dataset = edge_dataset.prefetch(10)

    return edge_dataset, batch_size, len(edges_to_exp), weight


def umap_loss(
    batch_size,
    negative_sample_rate,
    _a,
    _b,
    repulsion_strength=1.0,
):
    """
    Generate a keras-ccompatible loss function for UMAP loss

    Parameters
    ----------
    batch_size : int
        size of mini-batches
    negative_sample_rate : int
        number of negative samples per positive samples to train on
    _a : float
        distance parameter in embedding space
    _b : float float
        distance parameter in embedding space
    repulsion_strength : float, optional
        strength of repulsion vs attraction for cross-entropy, by default 1.0

    Returns
    -------
    loss : function
        loss function that takes in a placeholder (0) and the output of the keras network
    """

    @tf.function
    def loss(placeholder_y, embed_to_from):
        # split out to/from
        embedding_to, embedding_from, weights = tf.split(
            embed_to_from, num_or_size_splits=[2, 2, 1], axis=1
        )
        # embedding_to, embedding_from, weight = embed_to_from

        # get negative samples
        embedding_neg_to = tf.repeat(embedding_to, negative_sample_rate, axis=0)
        repeat_neg = tf.repeat(embedding_from, negative_sample_rate, axis=0)
        embedding_neg_from = tf.gather(
            repeat_neg, tf.random.shuffle(tf.range(tf.shape(repeat_neg)[0]))
        )

        #  distances between samples (and negative samples)
        distance_embedding = tf.concat(
            (
                tf.norm(embedding_to - embedding_from, axis=1),
                tf.norm(embedding_neg_to - embedding_neg_from, axis=1),
            ),
            axis=0,
        )

        # convert probabilities to distances
        probabilities_distance = convert_distance_to_probability(
            distance_embedding, _a, _b
        )

        # set true probabilities based on negative sampling
        probabilities_graph = tf.concat(
            (tf.ones(batch_size), tf.zeros(batch_size * negative_sample_rate)), axis=0,
        )
        probabilities = tf.concat(
            (tf.squeeze(weights), tf.zeros(batch_size * negative_sample_rate)), axis=0,
        )

        # compute cross entropy
        (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
            probabilities_graph,
            probabilities_distance,
            repulsion_strength=repulsion_strength,
        )

        return tf.reduce_mean(ce_loss)

    return loss

def compute_cross_entropy(
    probabilities_graph, probabilities_distance, EPS=1e-4, repulsion_strength=1.0
):
    """
    Compute cross entropy between low and high probability

    Parameters
    ----------
    probabilities_graph : array
        high dimensional probabilities
    probabilities_distance : array
        low dimensional probabilities
    probabilities : array
        edge weights + zeros
    EPS : float, optional
        offset to to ensure log is taken of a positive number, by default 1e-4
    repulsion_strength : float, optional
        strength of repulsion between negative samples, by default 1.0

    Returns
    -------
    attraction_term: tf.float32
        attraction term for cross entropy loss
    repellant_term: tf.float32
        repellant term for cross entropy loss
    cross_entropy: tf.float32
        cross entropy umap loss

    """
    # # cross entropy
    # attraction_term = (
    #         # probabilities * tf.math.log(tf.clip_by_value(probabilities, EPS, 1.0))
    #         - probabilities * tf.math.log(tf.clip_by_value(probabilities_distance, EPS, 1.0))
    # )
    # repellant_term = (
    #     -(1.0 - probabilities_graph)
    #     * tf.math.log(tf.clip_by_value(1.0 - probabilities_distance, EPS, 1.0))
    #     * repulsion_strength
    # )
    #
    # # balance the expected losses between atrraction and repel
    # CE = attraction_term + repellant_term
    # return attraction_term, repellant_term, CE
    # cross entropy
    attraction_term = -probabilities_graph * tf.math.log(
        tf.clip_by_value(probabilities_distance, EPS, 1.0)
    )
    repellant_term = (
            -(1.0 - probabilities_graph)
            * tf.math.log(tf.clip_by_value(1.0 - probabilities_distance, EPS, 1.0))
            * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def convert_distance_to_probability(distances, a=1.0, b=1.0):
    return 1.0 / (1.0 + a * distances ** (2 * b))


def boundary_wise_complex(centers, border_centers, n_neighbors):
    high_tree = KDTree(border_centers)
    from sklearn.utils import check_random_state
    random_state = check_random_state(None)

    fitting_data = np.concatenate((centers, border_centers), axis=0)
    knn_dists, knn_indices = high_tree.query(fitting_data, k=n_neighbors)
    knn_indices = knn_indices + len(centers)

    from umap.umap_ import fuzzy_simplicial_set
    bw_complex, sigmas, rhos = fuzzy_simplicial_set(
        X=fitting_data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    return bw_complex, sigmas, rhos


def fuzzy_complex(train_data, n_neighbors):
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
    # distance metric
    metric = "euclidean"

    from pynndescent import NNDescent
    # get nearest neighbors
    nnd = NNDescent(
        train_data,
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )

    from sklearn.utils import check_random_state
    from umap.umap_ import fuzzy_simplicial_set
    knn_indices, knn_dists = nnd.neighbor_graph
    random_state = check_random_state(None)
    complex, sigmas, rhos = fuzzy_simplicial_set(
        X=train_data,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )
    return complex, sigmas, rhos


def define_autoencoder(dims, n_components, units, encoder=None, decoder=None):
    if encoder is None:
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=dims),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            #     tf.keras.layers.Dense(units=1024, activation="relu"),
            #     tf.keras.layers.Dense(units=512, activation="relu"),
            tf.keras.layers.Dense(units=n_components),
        ])
    # define the decoder
    if decoder is None:
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_components)),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=np.product(dims), name="recon", activation=None),
            tf.keras.layers.Reshape(dims),

        ])
        return encoder, decoder


def define_model(dims, low_dims, encoder, decoder, temporal, stop_grad=False):
    # inputs
    to_x = tf.keras.layers.Input(shape=dims, name="to_x")
    from_x = tf.keras.layers.Input(shape=dims, name="from_x")
    weight = tf.keras.layers.Input(shape=(1,), name="weight")
    if not temporal:
        inputs = (to_x, from_x, weight)
    else:
        to_alpha = tf.keras.layers.Input(shape=(1, ), name="to_alpha")
        to_px = tf.keras.layers.Input(shape=(low_dims,), name="to_px")
        inputs = (to_x, from_x, to_alpha, to_px, weight)

    # parametric embedding
    embedding_to = encoder(to_x)
    embedding_from = encoder(from_x)

    if stop_grad:
        embedding_to_recon = decoder(tf.stop_gradient(embedding_to))
    else:
        # parametric reconstruction
        embedding_to_recon = decoder(embedding_to)


    embedding_to_recon = tf.keras.layers.Lambda(
        lambda x: x, name="reconstruction"
    )(embedding_to_recon)

    outputs = dict()
    outputs["reconstruction"] = embedding_to_recon

    embedding_to_from = tf.concat((embedding_to, embedding_from, weight), axis=1)
    embedding_to_from = tf.keras.layers.Lambda(lambda x: x, name="umap")(
        embedding_to_from
    )

    outputs["umap"] = embedding_to_from

    if temporal:
        embedding_diff = tf.concat((to_px, embedding_to, to_alpha), axis=1)
        embedding_diff = tf.keras.layers.Lambda(lambda x: x, name="temporal")(
            embedding_diff
        )
        outputs["temporal"] = embedding_diff

    parametric_model = tf.keras.Model(inputs=inputs, outputs=outputs,)
    return parametric_model


def define_losses(batch_size, n_epoch, tot_epochs, temporal):
    # compile models
    losses = {}
    loss_weights = {}

    # umap loss
    from umap.umap_ import find_ab_params
    min_dist = 0.1
    _a, _b = find_ab_params(1.0, min_dist)
    negative_sample_rate = 5

    umap_loss_fn = umap_loss(
        batch_size,
        negative_sample_rate,
        _a,
        _b,
    )

    losses["umap"] = umap_loss_fn
    loss_weights["umap"] = 1.0

    losses["reconstruction"] = tf.keras.losses.MeanSquaredError()
    loss_weights["reconstruction"] = 1.0

    if temporal:
        temporal_loss_fn = temporal_loss()
        losses["temporal"] = temporal_loss_fn
        loss_weights["temporal"] = 0.01

    # if temporal:
    #     ratio = n_epoch / float(tot_epochs)
    #     C = 1.0
    #     if ratio <= 0.3:
    #         C = 0
    #     elif ratio <= 0.5:
    #         C = 0.3
    #     elif ratio <= 0.8:
    #         C = 0.5
    #     temporal_loss_fn = temporal_loss()
    #     losses["temporal"] = temporal_loss_fn
    #     loss_weights["temporal"] = C

    return losses, loss_weights


def define_lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 5:
        lr *= 1e-1
    elif epoch > 10:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def construct_temporal_mixed_edge_dataset(
    X_input, old_graph_, new_graph_, n_epochs, batch_size, parametric_embedding,
        parametric_reconstruction, alpha, prev_embedding
):
    """
    Construct a tf.data.Dataset of edges, sampled by edge weight.

    Parameters
    ----------
    X : list, [X, DBP_samples]
        X : array, shape (n_samples, n_features)
            New data to be transformed.
        DBP_samples : array, shape(n_samples, n_features)
            distant border points to be transformed.
    old_graph_ : scipy.sparse.csr.csr_matrix
        Generated UMAP graph
    new_graph_: scipy.sparse.csr.csr_matrix
        Generated UMAP graph
    n_epochs : int
        # of epochs to train each edge
    batch_size : int
        batch size
    parametric_embedding : bool
        Whether the embedder is parametric or non-parametric
    parametric_reconstruction : bool
        Whether the decoder is parametric or non-parametric
    alpha : ndarray [n_samples]
    prev_embedding: [n_samples, 2(n_components)]
    """

    def gather_X(edge_to, edge_from, to_alpha, to_pe, weight):
        edge_to_batch = tf.gather(fitting_data, edge_to)
        edge_from_batch = tf.gather(fitting_data, edge_from)
        to_alpha_batch = tf.gather(alpha, to_alpha)
        to_pe_batch = tf.gather(prev_embedding, to_pe)

        outputs = {"umap": 0, "temporal": 0}
        if parametric_reconstruction:
            # add reconstruction to iterator output
            # edge_out = tf.concat([edge_to_batch, edge_from_batch], axis=0)
            outputs["reconstruction"] = edge_to_batch

        return (edge_to_batch, edge_from_batch, to_alpha_batch, to_pe_batch, weight), outputs

    train_data, centers, border_centers = X_input
    cp_num = len(train_data)
    centers_num = len(centers)
    bc_num = len(border_centers)
    fitting_data = np.concatenate((train_data, centers, border_centers), axis=0)
    alpha = np.expand_dims(alpha, axis=1)
    alpha = np.concatenate((alpha, np.zeros((centers_num + bc_num, 1))), axis=0)
    prev_embedding = np.concatenate((prev_embedding, np.zeros((centers_num + bc_num, 2))), axis=0)

    # get data from graph
    old_graph, old_epochs_per_sample, old_head, old_tail, old_weight, old_n_vertices = get_graph_elements(
        old_graph_, n_epochs
    )
    # get data from graph
    new_graph, new_epochs_per_sample, new_head, new_tail, new_weight, new_n_vertices = get_graph_elements(
        new_graph_, n_epochs
    )
    ## normalize two graphs
    new_head = new_head + cp_num
    new_tail = new_tail + cp_num

    # number of elements per batch for embedding
    if batch_size is None:
        # batch size randomly choose a number as batch_size if batch_size is None
        batch_size = 1000
    edges_to_exp = np.concatenate((np.repeat(old_head, old_epochs_per_sample.astype("int")),
                                   np.repeat(new_head, new_epochs_per_sample.astype("int"))), axis=0)
    edges_from_exp = np.concatenate((np.repeat(old_tail, old_epochs_per_sample.astype("int")),
                                   np.repeat(new_tail, new_epochs_per_sample.astype("int"))), axis=0)

    weight = np.concatenate((np.repeat(old_weight, old_epochs_per_sample.astype("int")),
                             np.repeat(new_weight, new_epochs_per_sample.astype("int"))), axis=0)

    # shuffle edges
    shuffle_mask = np.random.permutation(range(len(edges_to_exp)))
    edges_to_exp = edges_to_exp[shuffle_mask].astype(np.int64)
    edges_from_exp = edges_from_exp[shuffle_mask].astype(np.int64)
    weight = weight[shuffle_mask].astype(np.float64)
    # to_alpha = to_alpha[shuffle_mask].astype(np.float64)
    # to_pe = to_pe[shuffle_mask].astype(np.float64)
    weight = np.expand_dims(weight, axis=1)

    # create edge iterator
    edge_dataset = tf.data.Dataset.from_tensor_slices(
        (edges_to_exp, edges_from_exp, edges_to_exp, edges_to_exp, weight)
    )
    edge_dataset = edge_dataset.repeat()
    edge_dataset = edge_dataset.shuffle(10000)
    edge_dataset = edge_dataset.map(
        # gather_X, num_parallel_calls=tf.data.experimental.AUTOTUNE
        gather_X
    )
    edge_dataset = edge_dataset.batch(batch_size, drop_remainder=True)
    edge_dataset = edge_dataset.prefetch(10)

    return edge_dataset, batch_size, len(edges_to_exp), weight


def temporal_loss():
    @tf.function
    def loss(placeholder_y, x):
        to_px, embedding_to, to_alpha = tf.split(
            x, num_or_size_splits=[2, 2, 1], axis=1
        )
        to_alpha = tf.squeeze(to_alpha)
        to_alpha = to_alpha/(1+to_alpha)

        min_ = tf.minimum(to_px, embedding_to)
        max_ = tf.maximum(to_px, embedding_to)
        x_min = tf.reduce_min(min_[:, 0])
        y_min = tf.reduce_min(min_[:, 1])
        x_max = tf.reduce_max(max_[:, 0])
        y_max = tf.reduce_max(max_[:, 1])
        min_tot = tf.minimum(x_min, y_min)
        max_tot = tf.minimum(x_max, y_max)
        diff = (to_px-embedding_to)/(max_tot-min_tot)

        diff = tf.reduce_sum(tf.math.square(diff), axis=1)
        diff = tf.math.multiply(to_alpha, diff)
        return tf.reduce_mean(diff)
    return loss


def find_alpha(prev_data, train_data, n_neighbors):
    if prev_data is None:
        return np.zeros(len(train_data))
    # number of trees in random projection forest
    n_trees = min(64, 5 + int(round((train_data.shape[0]) ** 0.5 / 20.0)))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(train_data.shape[0]))))
    # distance metric
    metric = "euclidean"

    from pynndescent import NNDescent
    # get nearest neighbors
    nnd = NNDescent(
        train_data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    train_indices, _ = nnd.neighbor_graph
    prev_nnd = NNDescent(
        prev_data,
        n_neighbors=n_neighbors,
        metric="euclidean",
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True
    )
    prev_indices, _ = prev_nnd.neighbor_graph
    temporal_pres = np.zeros(len(train_data))
    for i in range(len(train_indices)):
        pres = np.intersect1d(train_indices[i], prev_indices[i])
        temporal_pres[i] = len(pres) / float(n_neighbors)
    return temporal_pres


def find_update_dist(prev_data, train_data, sigmas, rhos):
    if prev_data is None:
        return np.zeros(len(train_data))
    dists = np.linalg.norm(prev_data-train_data, axis=1)
    # weights = dists - rhos
    weights = dists
    index = np.bitwise_or(weights < 0, sigmas==0)
    weights = np.exp(-weights / sigmas)
    weights[index==True] = 1.0
    return weights



