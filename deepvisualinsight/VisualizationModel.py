'''
The visualization model definition class
'''
import tensorflow as tf
from tensorflow import keras


class ParametricModel(keras.Model):
    def __init__(self, encoder, decoder, optimizer, loss, loss_weights, temporal,
                 prev_trainable_variables=None):

        super(ParametricModel, self).__init__()
        self.encoder = encoder  # encoder part
        self.decoder = decoder  # decoder part
        self.optimizer = optimizer  # optimizer
        self.temporal = temporal

        self.loss = loss  # dict of 3 losses {"total", "umap", "reconstrunction", "regularization"}
        self.loss_weights = loss_weights  # weights for each loss (in total 3 losses)

        self.prev_trainable_variables = prev_trainable_variables  # weights for previous iteration
        self.e_var = None
        self.d_var = None

    def train_step(self, x):

        if self.temporal:
        #     # get one batch
            to_x, from_x, to_alpha, _, weight = x[0]

            # if self.e_var is None:
            #     self.e_var = [var for var in self.trainable_variables if "e_" in var.name]
            # if self.d_var is None:
            #     self.d_var = [var for var in self.trainable_variables if "d_" in var.name or "recon" in var.name]
        else:
            to_x, from_x, to_alpha, from_alpha, weight = x[0]

        # Forward pass
        with tf.GradientTape(persistent=True) as tape:

            # parametric embedding
            embedding_to = self.encoder(to_x)  # embedding for instance 1
            embedding_from = self.encoder(from_x)  # embedding for instance 1
            embedding_to_recon = self.decoder(embedding_to)  # reconstruct instance 1
            embedding_from_recon = self.decoder(embedding_from)  # reconstruct instance 1

            # concatenate embedding1 and embedding2 to prepare for umap loss
            embedding_to_from = tf.concat((embedding_to, embedding_from, tf.cast(weight, dtype=tf.float32, name=None)),
                                          axis=1)

            # reconstruction loss
            # reconstruct_loss = self.loss["reconstruction"](y_true=to_x, y_pred=embedding_to_recon) + \
            #                    self.loss["reconstruction"](y_true=from_x, y_pred=embedding_from_recon)

            reconstruct_loss = self.loss["reconstruction"](tf.cast(to_x, dtype=tf.float32), tf.cast(from_x, dtype=tf.float32), embedding_to_recon, embedding_from_recon, to_alpha, from_alpha)

            # umap loss
            umap_loss = self.loss["umap"](None, embed_to_from=embedding_to_from)  # w_(t-1), no gradient

            if self.temporal:
                # compute alpha bar
                alpha_mean = tf.cast(tf.reduce_mean(tf.stop_gradient(to_alpha)), dtype=tf.float32)
                prev_trainable_variables = self.prev_trainable_variables

                # embedding loss
                # embed_loss_to = self.loss["embedding_to"](_, embedding_to)
                # embed_loss_to_recon = self.loss["embedding_to_recon"](_, embedding_to_recon)
                #
                # final_grad_result_list = list()
                #
                # grad_e_var = tape.gradient(tf.reduce_mean(embed_loss_to, axis=0), self.e_var)
                # for i in range(len(self.e_var)):
                #     final_grad_result_list.append(tf.math.abs(tf.stop_gradient(grad_e_var[i])))
                # grad_d_var = tape.gradient(tf.reduce_mean(embed_loss_to_recon, axis=0), self.d_var)
                # for i in range(len(self.d_var)):
                #     final_grad_result_list.append(tf.math.abs(tf.stop_gradient(grad_d_var[i])))

                # L2 norm of w current - w for last epoch (subject model's epoch)
                if self.prev_trainable_variables is not None:
                # if len(weights_dict) != 0:
                    regularization_loss = self.loss["regularization"](w_prev=prev_trainable_variables,
                                                                      w_current=self.trainable_variables,
                                                                      to_alpha=alpha_mean)
                # dummy zero-loss if no previous epoch
                else:
                    prev_trainable_variables = [tf.stop_gradient(x) for x in self.trainable_variables]
                    regularization_loss = self.loss["regularization"](w_prev=prev_trainable_variables,
                                                                      w_current=self.trainable_variables,
                                                                      to_alpha=alpha_mean)
                    # aggregate loss, weighted average
                loss = tf.add(tf.add(tf.math.multiply(tf.constant(self.loss_weights["reconstruction"]), reconstruct_loss),
                                     tf.math.multiply(tf.constant(self.loss_weights["umap"]), umap_loss)),
                              tf.math.multiply(tf.constant(self.loss_weights["regularization"]), regularization_loss))
            else:
                loss = tf.add(tf.math.multiply(tf.constant(self.loss_weights["reconstruction"]), reconstruct_loss),
                           tf.math.multiply(tf.constant(self.loss_weights["umap"]), umap_loss))

        # Compute gradients
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        if self.temporal:
            return {"loss": loss, "umap": umap_loss, "reconstruction": reconstruct_loss,
                    "regularization": regularization_loss}
        else:
            return {"loss": loss, "umap": umap_loss, "reconstruction": reconstruct_loss}