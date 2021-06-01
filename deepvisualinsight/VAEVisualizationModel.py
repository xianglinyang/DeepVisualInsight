
import tensorflow as tf
from tensorflow import keras
from backend import *



class ParametricModel(keras.Model):
    def __init__(self, encoder, decoder, optimizer, 
                 loss, loss_weights,
                 prev_trainable_variables=None):
        
        super(ParametricModel, self).__init__()
        self.encoder = encoder # encoder part
        self.decoder = decoder # decoder part
        self.optimizer = optimizer # optimizer
        
        self.loss = loss # dict of 4 losses {"umap", "reconstrunction", "regularization", "KL"}
        self.loss_weights = loss_weights # weights for each loss (in total 4 losses)
        
        self.prev_trainable_variables = prev_trainable_variables # save weights for previous iteration

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=mean.shape)
        z = eps * tf.exp(logvar * .5) + mean # sample with reparameterization trick
        return mean, logvar, z

    def decode(self, z):
        logits = self.decoder(z)
        return logits
    
    def train_step(self, x):
        
        # get one batch 
        to_x, from_x, to_alpha, _, weight = x[0]

        # Forward pass
        with tf.GradientTape() as tape:
            # compute alpha bar
            alpha_mean = tf.cast(tf.reduce_mean(tf.stop_gradient(to_alpha)), dtype=tf.float32)
            
            # parametric embedding
            z_mean, z_log_var, embedding_to = self.encode(to_x)
            z_mean_from, z_log_var_from, embedding_from = self.encode(from_x)
            embedding_to_recon = self.decode(embedding_to)

            # concatenate embedding1 and embedding2 to prepare for umap loss
            embedding_to_from = tf.concat((embedding_to, embedding_from, tf.cast(weight, dtype=tf.float32, name=None)), axis=1)
            
            # kl loss
            kl_loss = self.loss['kl'](z_log_var = z_log_var, z_mean = z_mean)
            
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
            loss = tf.math.multiply(tf.constant(self.loss_weights["reconstruction"]), reconstruct_loss) + \
                  tf.math.multiply(tf.constant(self.loss_weights["umap"]), umap_loss) + \
                  tf.math.multiply(tf.constant(self.loss_weights["regularization"]), regularization_loss) + \
                  tf.math.multiply(tf.constant(self.loss_weights["kl"]), kl_loss)      

            
        # Compute gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))
        
        return {"loss": loss}