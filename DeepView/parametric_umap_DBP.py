import numpy as np
from umap import UMAP
from warnings import warn, catch_warnings, filterwarnings
from umap.umap_ import make_epochs_per_sample
from numba import TypingError
import os
from umap.spectral import spectral_layout
from sklearn.utils import check_random_state
import codecs, pickle
from sklearn.neighbors import KDTree
import tensorflow as tf

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


def make_balance_per_sample(edges_to_exp, edges_from_exp, tptp_edges_num, dbpdbp_edges_num, tpdbp_edges_num, tp_num):
    balance_per_sample = np.zeros(shape=(len(edges_to_exp)))
    dbpdbp_ratio = int(tptp_edges_num / dbpdbp_edges_num)
    tpdbp_ratio = int(tptp_edges_num / tpdbp_edges_num)
    # balance_per_sample[(edges_to_exp >= tp_num) & (edges_from_exp >= tp_num)] = 0  #dbpdbp)
    balance_per_sample[(edges_to_exp < tp_num) & (edges_from_exp >= tp_num)] = 1    #tpdbp
    balance_per_sample[(edges_to_exp >= tp_num) & (edges_from_exp < tp_num)] = 1   #dbptp
    #
    # balance_per_sample = np.zeros(shape=(len(edges_to_exp)))
    # balance_per_sample[(edges_to_exp < tp_num) & (edges_from_exp >= tp_num)] = 1

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
    #
    #
    # balance_per_sample = make_balance_per_sample(edges_to_exp, edges_from_exp,
    #                                               tptp_edges_num, dbpdbp_edges_num, tpdbp_edges_num, tp_num)
    #
    # edges_to_exp, edges_from_exp = (
    #     np.repeat(edges_to_exp, balance_per_sample.astype("int")),
    #     np.repeat(edges_from_exp, balance_per_sample.astype("int")),
    # )
    # weight = np.repeat(weight, balance_per_sample.astype("int"))


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


def umap_loss(
    batch_size,
    negative_sample_rate,
    _a,
    _b,
    edge_weights,
    parametric_embedding,
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
    edge_weights : array, (batch_size, 1)
        weights of all edges from sparse UMAP graph
    parametric_embedding : bool
        whether the embeddding is parametric or nonparametric
    repulsion_strength : float, optional
        strength of repulsion vs attraction for cross-entropy, by default 1.0

    Returns
    -------
    loss : function
        loss function that takes in a placeholder (0) and the output of the keras network
    """

    # if not parametric_embedding:
    #     # multiply loss by weights for nonparametric
    #     weights_tiled = np.tile(edge_weights, negative_sample_rate + 1)

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

        # if not parametric_embedding:
        #     ce_loss = ce_loss * weights_tiled

        return tf.reduce_mean(ce_loss)

    return loss

def tpdbp_loss():
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
    edge_weights : array, (batch_size, 1)
        weights of all edges from sparse UMAP graph
    parametric_embedding : bool
        whether the embeddding is parametric or nonparametric
    repulsion_strength : float, optional
        strength of repulsion vs attraction for cross-entropy, by default 1.0

    Returns
    -------
    loss : function
        loss function that takes in a placeholder (0) and the output of the keras network
    """

    # if not parametric_embedding:
    #     # multiply loss by weights for nonparametric
    #     weights_tiled = np.tile(edge_weights, negative_sample_rate + 1)

    @tf.function
    def loss(placeholder_y, weights):
        # split out to/from
        weights = tf.squeeze(weights)
        smallweights = (weights > 0.5)

        num = tf.math.reduce_sum(tf.cast(smallweights, tf.float64))
        return num
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
