import os
os.chdir('..')

import tensorflow as tf
from utils import *
from backend import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from evaluate import *
import gc
from scipy.special import softmax
import numpy as np
from tensorflow import keras

def define_autoencoder(dims, n_components, units, encoder=None, decoder=None):
    if encoder is None:
        encoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=dims),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=n_components),
        ])
    # define the decoder
    if decoder is None:
        decoder = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(n_components, )),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=units, activation="relu"),
            tf.keras.layers.Dense(units=np.product(dims), name="recon", activation=None),
            tf.keras.layers.Reshape(dims),

        ])
        return encoder, decoder
    
    
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
    if epoch > 10:
        lr *= 1e-1
    elif epoch > 20:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def regularize_loss():
    '''
    Add temporal regularization L2 loss on weights
    '''
    @tf.function
    def loss(w_prev, w_current, to_alpha):
        assert len(w_prev) == len(w_current)
        # multiple layers of weights, need to add them up
        for j in range(len(w_prev)): 
            diff = tf.reduce_sum(tf.math.square(w_current[j] - w_prev[j]))
            diff = tf.math.multiply(to_alpha, diff)
            if j == 0:
                alldiff = tf.reduce_mean(diff)
            else:
                alldiff += tf.reduce_mean(diff)
        return alldiff
    
    return loss


def define_losses(batch_size, n_epoch, tot_epochs):
    '''
    Define customized loss
    '''
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

    regularize_loss_fn = regularize_loss()
    losses["regularization"] = regularize_loss_fn
    loss_weights["regularization"] = 1.0

    return losses, loss_weights



class ParametricModel(keras.Model):
    def __init__(self, encoder, decoder, optimizer, loss, loss_weights,
                 prev_trainable_variables=None):
        
        super(ParametricModel, self).__init__()
        self.encoder = encoder # encoder part
        self.decoder = decoder # decoder part
        self.optimizer = optimizer # optimizer
        
        self.loss = loss # dict of 3 losses {"umap", "reconstrunction", "regularization"}
        self.loss_weights = loss_weights # weights for each loss (in total 3 losses)
        
        self.prev_trainable_variables = prev_trainable_variables # save weights for previous iteration
        
    def train_step(self, x):
        
        # get one batch 
        to_x, from_x, to_alpha, _, weight = x[0]

        # Forward pass
        with tf.GradientTape() as tape:
            # compute alpha bar
            alpha_mean = tf.cast(tf.reduce_mean(tf.stop_gradient(to_alpha)), dtype=tf.float32)
            
            # parametric embedding
            embedding_to = encoder(to_x) # embedding for instance 1
            embedding_from = encoder(from_x) # embedding for instance 1
            embedding_to_recon = decoder(embedding_to) # reconstruct instance 1
            
            # concatenate embedding1 and embedding2 to prepare for umap loss
            embedding_to_from = tf.concat((embedding_to, embedding_from, tf.cast(weight, dtype=tf.float32, name=None)), axis=1)
            
            # reconstruction loss
            reconstruct_loss = self.loss["reconstruction"](y_true=to_x, y_pred=embedding_to_recon)
            
            # umap loss
            umap_loss = self.loss["umap"](_, embed_to_from=embedding_to_from)  # w_(t-1), no gradient
            
            # regularization loss
            self.prev_trainable_variables = [tf.stop_gradient(x) for x in self.trainable_variables] 
                
            regularization_loss = self.loss["regularization"](w_prev=self.prev_trainable_variables, 
                                                              w_current=self.trainable_variables, 
                                                              to_alpha=alpha_mean)
            # aggregate loss, weighted average
            loss = tf.add(tf.add(tf.math.multiply(tf.constant(self.loss_weights["reconstruction"]), reconstruct_loss),
                          tf.math.multiply(tf.constant(self.loss_weights["umap"]), umap_loss)),
                          tf.math.multiply(tf.constant(self.loss_weights["regularization"]), regularization_loss))

#         print(loss)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return {"loss": loss}
    
def get_proj_model(epoch_id):
    '''
    get the encoder of epoch_id
    :param epoch_id: int
    :return: encoder of epoch epoch_id
    '''
    if temporal:
        encoder_location = os.path.join('deepvisualinsight/models/', "Epoch_{:d}".format(epoch_id), "encoder_temporal")
    else:
        encoder_location = os.path.join('deepvisualinsight/models/', "Epoch_{:d}".format(epoch_id), "encoder")
    if os.path.exists(encoder_location):
        encoder = tf.keras.models.load_model(encoder_location)
        return encoder
    else:
        print("Error! Projection function has not been initialized! Pls first visualize all.")
        return None
    
if __name__ == "__main__":
    
    # fix those parameters for debugging
    repr_num = 512
    dims = (repr_num,)
    n_components = 2
    batch_size = 200
    n_epoch = 2
    temporal = True
    low_dims = 2
    neurons = repr_num / 2
    
    # define model
    encoder, decoder = define_autoencoder(dims, n_components, neurons)
    optimizer = tf.keras.optimizers.Adam(1e-3)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            min_delta=10 ** -2,
            patience=8,
            verbose=1,
        ),
        tf.keras.callbacks.LearningRateScheduler(define_lr_schedule),
    ]

    # define loss
    losses, loss_weights = define_losses(200, n_epoch=n_epoch, tot_epochs=200-1+1)
    parametric_model = ParametricModel(encoder=encoder, decoder=decoder, optimizer=optimizer,
                                      loss=losses, loss_weights=loss_weights,
                                      prev_trainable_variables=None)
    
    parametric_model.compile(loss=losses, loss_weights=loss_weights)
    
    # load data
    train_centers_loc = os.path.join('deepvisualinsight/models', "Epoch_{:d}".format(n_epoch), "train_centers.npy")
    border_centers_loc = os.path.join('deepvisualinsight/models', "Epoch_{:d}".format(n_epoch), "border_centers.npy")
    train_data_loc = os.path.join('deepvisualinsight/models', "Epoch_{:d}".format(n_epoch), "train_data.npy")

    train_centers = np.load(train_centers_loc)
    border_centers = np.load(border_centers_loc)
    train_data = np.load(train_data_loc)

    # prepare data 
    complex, sigmas, rhos = fuzzy_complex(train_data, 15)
    bw_complex, _, _ = boundary_wise_complex(train_centers, border_centers, 15)

    # load prev iteration's GAP
    prev_data_loc = os.path.join('deepvisualinsight/models/', "Epoch_{:d}".format(n_epoch-1), "train_data.npy")
    if os.path.exists(prev_data_loc) and 0 != n_epoch:
        prev_data = np.load(prev_data_loc)
    else:
        prev_data = None
    if prev_data is None:
        prev_embedding = np.zeros((len(train_data), self.low_dims))
    else:
        encoder = get_proj_model(n_epoch-1)
        prev_embedding = encoder(prev_data).cpu().numpy()
    
    
    # construct data
    alpha = find_alpha(prev_data, train_data, n_neighbors=15)
    alpha[alpha < 0.5] = 0.0 # alpha >=0.5 is convincing
    update_dists = find_update_dist(prev_data, train_data, sigmas, rhos)
    update_dists[update_dists < 0.05] = 0.0
    alpha = alpha*update_dists
    (
        edge_dataset,
        batch_size,
        n_edges,
        edge_weight,
    ) = construct_temporal_mixed_edge_dataset(
        (train_data, train_centers, border_centers),
        complex,
        bw_complex,
        20,
        batch_size,
        parametric_embedding=True,
        parametric_reconstruction=True,
        alpha=alpha,
        prev_embedding=prev_embedding
    )

    steps_per_epoch = int(
        n_edges / batch_size / 10
    )
    
    # model training
    history = parametric_model.fit(
        edge_dataset,
        epochs=200,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks,
        max_queue_size=100,
    )
    