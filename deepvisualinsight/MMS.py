import os
import time

import numpy as np
import pyasn1_modules.rfc6031
import tensorflow as tf
from deepvisualinsight.utils import *
from deepvisualinsight.backend import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from deepvisualinsight.evaluate import *
import gc
import pandas as pd
from scipy.special import softmax
from scipy.spatial.distance import cdist
import deepvisualinsight.utils_advanced as utils_advanced
from deepvisualinsight.VisualizationModel import ParametricModel


class MMS:
    def __init__(self, content_path, model_structure, epoch_start, epoch_end, period, repr_num, class_num, classes,
                 low_dims=2, cmap="tab10", resolution=100, neurons=None, batch_size=1000, verbose=1, split=-1, attack_device="cpu",
                 advance_border_gen=True, alpha=0.7, withoutB=False,    # boundary complex
                 attention=True, temperature=None,                      # reconstruction loss
                 temporal=False, transfer_learning=True, step3=False):  # temporal

        '''
        This class contains the model management system (super DB) and provides
        several DVI user interface for dimension reduction and inverse projection function
        This class serves as a backend for DeepVisualInsight plugin.

        Parameters
        ----------
        content_path : str
            the path to model, training data and testing data
        model_structure : torch.function
            the structure of subject model
        epoch_start : int
            the epoch id that serves as start of visualization
        epoch_end : int
            the epoch id that serves as the end of visualization
        period : int
            seletive choose epoch to visualize
        repr_num : int
            the dimension of embeddings
        class_num : int
            the length of classification labels
        classes	: list, tuple of str
            All classes that the classifier uses as a list/tuple of strings.
        temperature: float
            the temperature sharpening parameter for Reconstruction loss(gradient calculation)
        low_dims: int
            the number of dimensions in low dimensional space
        cmap : str, by default 'tab10', should support more classes in the future
            Name of the colormap to use for visualization.
            The number of distinguishable colors should correspond to class_num.
            See here for the names: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        resolution : int, by default 100
            Resolution of the classification boundary plot.
        neurons : int
            the number of units inside each layer of autoencoder
        temporal: boolean, by default False
            choose whether to add temporal loss or not
        transfer_learning: boolean, by default True
            choose whether to use transfer learning
        step3: whether to use step3 temporal loss, by default False(step2 instead)
        batch_size: int, by default 1000
            the batch size to train autoencoder
        attention: bool, by default True
            whether to use attention for reconstruction loss(increase inv accu)
        verbose : int, by default 1
        split: int, by default -1
            number of layers for feature function
        advance_border_gen : boolean, by default True
            whether to use advance adversarial attack method for border points generation
        alpha: float
            lower bound for, new_image = alpha*m1+(1-alpha)*m2
            upper bound in paper, but they are same
        withoutB: boolean, by default False
            whether to add boundary preserving property, for baseline comparsion
        attack_device: str, by default "cpu"
            the device the perform adversatial attack
        '''
        self.model = model_structure
        self.visualization_models = None
        self.subject_models = None
        self.content_path = content_path
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.period = period
        self.training_data = None
        self.data_epoch_index = None
        self.testing_data = None
        self.repr_num = repr_num
        self.low_dims = low_dims
        self.cmap = plt.get_cmap(cmap)
        self.resolution = resolution
        self.class_num = class_num
        self.classes = classes
        self.temporal = temporal
        self.step3 = step3
        self.temperature = temperature
        self.transfer_learning = transfer_learning
        self.batch_size = batch_size
        self.verbose = verbose
        self.split = split
        # TODO depercate advanced attack, set it as default
        self.advance_border_gen = advance_border_gen
        self.alpha = alpha
        self.withoutB = withoutB
        self.device = torch.device(attack_device)
        self.attention = attention
        # TODO change tensorflow to pytorch
        if len(tf.config.list_physical_devices('GPU')) > 0:
            # self.tf_device = tf.config.list_physical_devices('GPU')[0]
            for d in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(d, True)
        # else:
        #     self.tf_device = tf.config.list_physical_devices('CPU')[0]
        if neurons is None:
            self.neurons = self.repr_num / 2
        else:
            self.neurons = neurons
        self.load_content()
        self.check_sanity()

    def check_sanity(self):
        """
        check whether all inputs are legal
            attention with temperature
            boundary construction
            temporal loss
        :return: None
        """
        if self.attention and self.temperature is None:
            raise Exception("illegal temperature. Pls check input of temperature!")
        if self.temporal and not self.transfer_learning:
            raise Exception("temporal loss should be put on top of transfer learning!")

    def load_content(self):
        '''
        load training dataset and testing dataset
        '''
        if not os.path.exists(self.content_path):
            sys.exit("Dir not exists!")

        self.training_data_path = os.path.join(self.content_path, "Training_data")
        self.training_data = torch.load(os.path.join(self.training_data_path, "training_dataset_data.pth"))
        self.training_labels = torch.load(os.path.join(self.training_data_path, "training_dataset_label.pth"))
        self.testing_data_path = os.path.join(self.content_path, "Testing_data")
        self.testing_data = torch.load(os.path.join(self.testing_data_path, "testing_dataset_data.pth"))
        self.testing_labels = torch.load(os.path.join(self.testing_data_path, "testing_dataset_label.pth"))

        self.model_path = os.path.join(self.content_path, "Model")
        if self.verbose > 0:
            print("Finish loading content!")

    #################################################### Trainer ####################################################
    def data_preprocessing(self):
        '''
        preprocessing data. This process includes find_border_points and find_border_centers
        save data for later training
        '''
        time_borders_gen = list()
        for n_epoch in range(self.epoch_start, self.epoch_end+1, self.period):
            index_file = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "index.json")
            index = load_labelled_data_index(index_file)
            training_data = self.training_data[index]
            training_labels = self.training_labels[index]

            # make it possible to choose a subset of testing data for testing
            test_index_file = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "test_index.json")
            if os.path.exists(test_index_file):
                test_index = load_labelled_data_index(test_index_file)
            else:
                test_index = range(len(self.testing_data))
            testing_data = self.testing_data[test_index]

            model_location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "subject_model.pth")
            self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
            self.model = self.model.to(self.device)
            self.model.eval()

            repr_model = torch.nn.Sequential(*(list(self.model.children())[:self.split]))

            n_clusters = math.floor(len(index) / 10)

            if self.advance_border_gen:
                t0 = time.time()
                training_data = training_data.to(self.device)
                confs = batch_run(self.model, training_data, 10)
                preds = np.argmax(confs, axis=1).squeeze()
                num_adv_eg = int(len(training_data)/10)
                # TODO refactor to one utils.py file, remove utils_advanced
                border_points, curr_samples, tot_num = utils_advanced.get_border_points(model=self.model,
                                                                                     input_x=training_data,
                                                                                     confs=confs,
                                                                                     predictions=preds,
                                                                                     device=self.device,
                                                                                     alpha=self.alpha,
                                                                                     num_adv_eg=num_adv_eg,
                                                                                     # num_cls=10,
                                                                                     lambd=0.05,
                                                                                     verbose=0)
                t1 = time.time()
                time_borders_gen.append(round(t1 - t0, 4))

                # get gap layer data
                border_points = border_points.to(self.device)
                border_centers = batch_run(repr_model, border_points, self.repr_num)
                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "advance_border_centers.npy")
                np.save(location, border_centers)

                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "ori_advance_border_centers.npy")
                np.save(location, border_points.cpu().numpy())

                confs = batch_run(self.model, border_points, 10)
                border_cls = np.argmax(confs, axis=1).squeeze()
                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "advance_border_labels.npy")
                np.save(location, np.array(border_cls))
            else:
                # soon to be depercated in the future
                # border points gen
                t0 = time.time()
                border_points = get_border_points(training_data, training_labels, self.model, self.device)

                # border clustering
                border_points = torch.from_numpy(border_points)
                border_points = border_points.to(self.device)
                border_representation = batch_run(repr_model, border_points, self.repr_num)
                border_centers = clustering(border_representation, n_clusters, verbose=0)
                t1 = time.time()
                time_borders_gen.append(round(t1 - t0, 4))

                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "border_centers.npy")
                np.save(location, border_centers)

                index = cdist(border_centers, border_representation, 'euclidean').argmax(-1)
                ori_border_points = border_points.cpu().numpy()[index]
                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "ori_border_centers.npy")
                np.save(location, ori_border_points)

            # training data clustering
            data_pool = self.training_data
            data_pool = data_pool.to(self.device)
            data_pool_representation = batch_run(repr_model, data_pool, self.repr_num)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_data.npy")
            np.save(location, data_pool_representation)

            # test data
            test_data = testing_data.to(self.device)
            test_data_representation = batch_run(repr_model, test_data, self.repr_num)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "test_data.npy")
            np.save(location, test_data_representation)

            if self.verbose > 0:
                print("Finish data preprocessing for Epoch {:d}...".format(n_epoch))
        print(
            "Average time for generate border points: {:.4f}".format(sum(time_borders_gen) / len(time_borders_gen)))

        # save result
        save_dir = os.path.join(self.model_path,  "time.json")
        if not os.path.exists(save_dir):
            evaluation = dict()
        else:
            f = open(save_dir, "r")
            evaluation = json.load(f)
            f.close()
        evaluation["data_gene"] = round(sum(time_borders_gen) / len(time_borders_gen), 3)
        with open(save_dir, 'w') as f:
            json.dump(evaluation, f)

        self.model = self.model.to(self.device)

    def eval_keep_B(self, name=""):
        # evaluate keep_B metric
        t_s = time.time()
        epoch_num = int((self.epoch_end - self.epoch_start) / self.period + 1)
        for n_epoch in range(self.epoch_start, self.epoch_end + 1, self.period):
            save_dir = os.path.join(self.model_path, "Epoch_{}".format(n_epoch), "evaluation{}.json".format(name))
            if os.path.exists(save_dir):
                with open(save_dir, "r") as f:
                    evaluation = json.load(f)
            else:
                evaluation = {}
            evaluation['keep_B_train'] = self.keep_B_train(n_epoch)
            evaluation['keep_B_test'] = self.keep_B_test(n_epoch)
            evaluation['keep_B_bound'] = self.keep_B_boundary(n_epoch)
            with open(save_dir, 'w') as f:
                json.dump(evaluation, f)
        t_e = time.time()
        if self.verbose > 0 :
            print("Average evaluation time for 1 epoch is {:.2f} seconds".format((t_e-t_s) / epoch_num))

    def save_evaluation(self, eval=False, name=""):
        # evaluation information
        t_s = time.time()
        epoch_num = int((self.epoch_end - self.epoch_start) / self.period + 1)
        for n_epoch in range(self.epoch_start, self.epoch_end + 1, self.period):
            save_dir = os.path.join(self.model_path, "Epoch_{}".format(n_epoch), "evaluation{}.json".format(name))
            if os.path.exists(save_dir):
                with open(save_dir, "r") as f:
                    evaluation = json.load(f)
            else:
                evaluation = {}
            evaluation['nn_train_15'] = self.proj_nn_perseverance_knn_train(n_epoch, 15)
            evaluation['nn_test_15'] = self.proj_nn_perseverance_knn_test(n_epoch, 15)
            evaluation['bound_train_15'] = self.proj_boundary_perseverance_knn_train(n_epoch, 15)
            evaluation['bound_test_15'] = self.proj_boundary_perseverance_knn_test(n_epoch, 15)
            evaluation['keep_B_train'] = self.keep_B_train(n_epoch)
            evaluation['keep_B_test'] = self.keep_B_test(n_epoch)
            evaluation['keep_B_bound'] = self.keep_B_boundary(n_epoch)

            # for paper evaluation
            if eval:
                evaluation['nn_train_10'] = self.proj_nn_perseverance_knn_train(n_epoch, 10)
                evaluation['nn_test_10'] = self.proj_nn_perseverance_knn_test(n_epoch, 10)
                evaluation['bound_train_10'] = self.proj_boundary_perseverance_knn_train(n_epoch, 10)
                evaluation['bound_test_10'] = self.proj_boundary_perseverance_knn_test(n_epoch, 10)

                evaluation['nn_train_20'] = self.proj_nn_perseverance_knn_train(n_epoch, 20)
                evaluation['nn_test_20'] = self.proj_nn_perseverance_knn_test(n_epoch, 20)
                evaluation['bound_train_20'] = self.proj_boundary_perseverance_knn_train(n_epoch, 20)
                evaluation['bound_test_20'] = self.proj_boundary_perseverance_knn_test(n_epoch, 20)
            print("finish proj eval for Epoch {}".format(n_epoch))

            evaluation['inv_acc_train'] = self.inv_accu_train(n_epoch)
            evaluation['inv_acc_test'] = self.inv_accu_test(n_epoch)
            evaluation['inv_conf_train'] = self.inv_conf_diff_train(n_epoch)
            evaluation['inv_conf_test'] = self.inv_conf_diff_test(n_epoch)
            print("finish inv eval for Epoch {}".format(n_epoch))

            # record time to project and inverse testing data
            test_data = self.get_epoch_test_repr_data(n_epoch)
            test_len = len(test_data)

            proj = self.get_proj_model(n_epoch)
            t0 = time.time()
            test_embedded = proj(test_data)
            t1 = time.time()
            del proj
            gc.collect()

            inv = self.get_inv_model(n_epoch)
            t2 = time.time()
            _ = inv(test_embedded)
            t3 = time.time()
            del inv
            gc.collect()

            evaluation["time_test_proj"] = round(t1-t0, 3)
            evaluation["time_test_inv"] = round(t3-t2, 3)
            evaluation["test_len"] = test_len

            # evaluation["temporal_train"] = self.proj_temporal_perseverance_train(15)
            # evaluation["temporal_test"] = self.proj_temporal_perseverance_test(15)

            # subject model train/test accuracy
            evaluation['acc_train'] = self.training_accu(n_epoch)
            evaluation['acc_test'] = self.testing_accu(n_epoch)
            print("finish subject model eval for Epoch {}".format(n_epoch))

            with open(save_dir, 'w') as f:
                json.dump(evaluation, f)
        t_e = time.time()
        if self.verbose > 0 :
            print("Average evaluation time for 1 epoch is {:.2f} seconds".format((t_e-t_s) / epoch_num))

    def prepare_visualization_for_all(self, encoder_in=None, decoder_in=None):
        """
        conduct transfer learning to save the visualization model for each epoch
        """
        dims = (self.repr_num,)
        n_components = 2
        batch_size = self.batch_size
        encoder, decoder = define_autoencoder(dims, n_components, self.neurons)
        if encoder_in is not None:
            encoder = encoder_in
        if decoder_in is not None:
            decoder = decoder_in
        optimizer = tf.keras.optimizers.Adam()

        weights_dict = {}
        losses, loss_weights = define_losses(batch_size, self.temporal, self.step3, self.withoutB, self.attention)
        parametric_model = ParametricModel(encoder, decoder, optimizer, losses, loss_weights, self.temporal, self.step3,
                                           self.withoutB, self.attention, self.batch_size, prev_trainable_variables=None)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=10 ** -2,
                patience=8,
                verbose=1,
            ),
            tf.keras.callbacks.LearningRateScheduler(define_lr_schedule),
            tf.keras.callbacks.LambdaCallback(on_train_end=lambda logs: weights_dict.update(
                {'prev': [tf.identity(tf.stop_gradient(x)) for x in parametric_model.trainable_weights]})),
        ]
        t0 = time.time()
        for n_epoch in range(self.epoch_start, self.epoch_end+1, self.period):
            if not self.transfer_learning:
                encoder, decoder = define_autoencoder(dims, n_components, self.neurons)
                if encoder_in is not None:
                    encoder = encoder_in
                if decoder_in is not None:
                    decoder = decoder_in
                parametric_model = ParametricModel(encoder, decoder, optimizer, losses, loss_weights,self.temporal,
                                                   self.step3, self.withoutB, self.attention, self.batch_size,
                                                   prev_trainable_variables=None)
            parametric_model.compile(
                optimizer=optimizer, loss=losses, loss_weights=loss_weights,
            )

            if self.advance_border_gen:
                border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch),
                                                  "advance_border_centers.npy")
            else:
                border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch),
                                                  "border_centers.npy")
            train_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_data.npy")

            try:
                train_data = np.load(train_data_loc)
                current_index = self.get_epoch_index(n_epoch)
                train_data = train_data[current_index]
            except Exception as e:
                print("no train data saved for Epoch {}".format(n_epoch))
                continue
            try:
                border_centers = np.load(border_centers_loc)
            except Exception as e:
                print("no border points saved for Epoch {}".format(n_epoch))
                continue

            # attention/temporal/complex
            if self.withoutB:
                all_d = np.concatenate((train_data, border_centers), axis=0)
                complex, sigmas, rhos = fuzzy_complex(all_d, 15)
                model_location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "subject_model.pth")
                self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
                self.model = self.model.to(self.device)
                self.model.eval()

                model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))
                model = model.to(self.device)
                model = model.eval()
                alpha = get_alpha(model, all_d, temperature=self.temperature, device=self.device, verbose=1)
                if self.temporal:
                    prev_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch-self.period), "train_data.npy")
                    if self.advance_border_gen:
                        prev_border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch),
                                                          "advance_border_centers.npy")
                    else:
                        prev_border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch),
                                                          "border_centers.npy")
                    if os.path.exists(prev_data_loc) and self.epoch_start != n_epoch:
                        prev_data = np.load(prev_data_loc)
                        prev_index = self.get_epoch_index(n_epoch-self.period)
                        prev_data = prev_data[prev_index]
                        prev_border = np.load(prev_border_centers_loc)
                    else:
                        prev_data = None
                        prev_border = None
                    if prev_data is not None:
                        prev_data = np.concatenate((prev_data, prev_border), axis=0)
                    n_rate = find_neighbor_preserving_rate(prev_data, all_d, n_neighbors=15)
                    (
                        edge_dataset,
                        batch_size,
                        n_edges,
                        edge_weight,
                    ) = construct_temporal_edge_dataset(
                        all_d,
                        complex,
                        10,
                        batch_size,
                        n_rate=n_rate,
                        alpha=alpha,
                    )

                else:
                    (
                        edge_dataset,
                        batch_size,
                        n_edges,
                        edge_weight,
                    ) = construct_edge_dataset(
                        all_d,
                        complex,
                        10,
                        batch_size,
                        alpha=alpha,
                    )
            else:
                complex, sigmas, rhos = fuzzy_complex(train_data, 15)
                bw_complex, _, _ = boundary_wise_complex(train_data, border_centers, 15)
                fitting_data = np.concatenate((train_data, border_centers), axis=0)

                model_location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "subject_model.pth")
                self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
                self.model = self.model.to(self.device)
                self.model.eval()

                model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))
                model = model.to(self.device)
                model = model.eval()
                alpha = get_alpha(model, fitting_data, temperature=self.temperature, device=self.device, verbose=1)
                del model
                gc.collect()
                if not self.temporal:
                    (
                        edge_dataset,
                        _batch_size,
                        n_edges,
                        _edge_weight,
                    ) = construct_mixed_edge_dataset(
                        (train_data, border_centers),
                        complex,
                        bw_complex,
                        10,
                        batch_size,
                        alpha,
                    )
                else:
                    prev_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch-self.period), "train_data.npy")
                    if os.path.exists(prev_data_loc) and self.epoch_start != n_epoch:
                        prev_data = np.load(prev_data_loc)
                        prev_index = self.get_epoch_index(n_epoch-self.period)
                        prev_data = prev_data[prev_index]
                    else:
                        prev_data = None
                    # if prev_data is None:
                    #     prev_embedding = np.zeros((len(train_data), self.low_dims))
                    # else:
                    #     encoder_ = self.get_proj_model(n_epoch-self.period)
                    #     prev_embedding = encoder_(prev_data).cpu().numpy()
                    #     del encoder_
                    #     gc.collect()
                    n_rate = find_neighbor_preserving_rate(prev_data, train_data, n_neighbors=15)

                    (
                        edge_dataset,
                        batch_size,
                        n_edges,
                        edge_weight,
                    ) = construct_temporal_mixed_edge_dataset(
                        (train_data, border_centers),
                        complex,
                        bw_complex,
                        10,
                        batch_size,
                        n_rate=n_rate,
                        alpha=alpha,
                    )

            steps_per_epoch = int(
                n_edges / batch_size / 10
            )
            # create embedding
            _ = parametric_model.fit(
                edge_dataset,
                epochs=200, # a large value, because we have early stop callback
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                max_queue_size=100,
            )
            # save for later use
            parametric_model.prev_trainable_variables = weights_dict["prev"]
            flag = ""   # record boundary complex and attention
            if self.withoutB:
                flag += "_withoutB"
            elif self.advance_border_gen:
                flag += "_advance"
            if self.attention:
                flag += "_A"

            if self.temporal:
                if self.step3:
                    encoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "encoder_temporal3" + flag))
                    decoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "decoder_temporal3" + flag))
                else:
                    encoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "encoder_temporal"+flag))
                    decoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "decoder_temporal"+flag))
            elif self.transfer_learning:
                encoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "encoder"+flag))
                decoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "decoder"+flag))
            else:
                encoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "encoder_independent"+flag))
                decoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "decoder_independent"+flag))

            if self.verbose > 0:
                print("save visualized model for Epoch {:d}".format(n_epoch))
        t1 = time.time()
        if self.verbose > 0:
            print("Average time for training visualization model: {:.4f}".format(
                (t1 - t0) / int((self.epoch_end - self.epoch_start) / self.period + 1)))
        # save result
        save_dir = os.path.join(self.model_path,  "time.json")
        if not os.path.exists(save_dir):
            evaluation = dict()
        else:
            f = open(save_dir, "r")
            evaluation = json.load(f)
            f.close()
        if self.withoutB:
            time_label = "vis_model_ParametricUmap"
        elif not self.transfer_learning:
            time_label = "vis_model_NT"
        elif not self.temporal:
            time_label = "vis_model_T"
        elif not self.step3:
            time_label = "vis_model_step2"
        else:
            time_label = "vis_model_step3"
        evaluation[time_label] = round(
            (t1 - t0) / int((self.epoch_end - self.epoch_start) / self.period + 1), 3)
        with open(save_dir, 'w') as f:
            json.dump(evaluation, f)

    ################################################ Backend APIs ################################################
    def get_proj_model(self, epoch_id):
        '''
        get the encoder of epoch_id
        :param epoch_id: int
        :return: encoder of epoch epoch_id
        '''
        flag = ""
        if self.withoutB:
            flag += "_withoutB"
        elif self.advance_border_gen:
            flag = "_advance"
        if self.attention:
            flag += "_A"

        if self.temporal:
            if self.step3:
                encoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id),
                                                "encoder_temporal3" + flag)
            else:
                encoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "encoder_temporal"+flag)
        elif self.transfer_learning:
            encoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "encoder"+flag)
        else:
            encoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "encoder_independent"+flag)
        if os.path.exists(encoder_location):
            encoder = tf.keras.models.load_model(encoder_location)
            if self.verbose > 0:
                print("Keras encoder model loaded from {}".format(encoder))
            return encoder
        else:
            print("Error! Projection function has not been initialized! Pls first visualize all.")
            return None

    def get_inv_model(self, epoch_id):
        '''
        get the decoder of epoch_id
        :param epoch_id: int
        :return: decoder model of epoch_id
        '''
        flag = ""
        if self.withoutB:
            flag += "_withoutB"
        elif self.advance_border_gen:
            flag += "_advance"
        if self.attention:
            flag += "_A"

        if self.temporal:
            if self.step3:
                decoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id),
                                                "decoder_temporal3" + flag)
            else:
                decoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "decoder_temporal"+flag)
        elif self.transfer_learning:
            decoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "decoder"+flag)
        else:
            decoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "decoder_independent"+flag)
        if os.path.exists(decoder_location):
            decoder = tf.keras.models.load_model(decoder_location)
            if self.verbose > 0:
                print("Keras encoder model loaded from {}".format(decoder))
            return decoder
        else:
            print("Error! Inverse function has not been initialized! Pls first visualize all.")
            return None

    def batch_project(self, data, epoch_id):
        '''
        batch project data to 2D space
        :param data: numpy.ndarray
        :param epoch_id:
        :return: embedding numpy.ndarray
        '''
        encoder = self.get_proj_model(epoch_id)
        if encoder is None:
            return None
        else:
            # TODO : in case memory limitation, need to be separated into minibatch and run for final result
            embedding = encoder(data).cpu().numpy()
            return embedding

    def individual_project(self, data, epoch_id):
        '''
        project a data to 2D space
        :param data: numpy.ndarray
        :param epoch_id: int
        :return: embedding numpy.ndarray
        '''
        encoder = self.get_proj_model(epoch_id)
        if encoder is None:
            return None
        else:
            data = np.expand_dims(data, axis=0)
            embedding = encoder(data).cpu().numpy()
            return embedding.squeeze()

    def individual_inverse(self, data, epoch_id):
        """
        map a 2D point back into high dimensional space
        :param data: ndarray, (1, 2)
        :param epoch_id: num of epoch
        :return: high dim representation, numpy.ndarray
        """
        decoder = self.get_inv_model(epoch_id)
        if decoder is None:
            return None
        else:
            data = np.expand_dims(data, axis=0)
            representation_data = decoder(data).cpu().numpy()
            return representation_data.squeeze()

    def batch_inverse(self, data, epoch_id):
        """
        map 2D points back into high dimensional space
        :param data: ndarray, (n, 2)
        :param epoch_id: num of epoch
        :return: high dim representation, numpy.ndarray
        """
        decoder = self.get_inv_model(epoch_id)
        if decoder is None:
            return None
        else:
            # TODO separate into minibatch
            representation_data = decoder(data).cpu().numpy()
            return representation_data

    def get_incorrect_predictions(self, epoch_id, data, targets):
        '''
        get if the prediction of classifier is true or not
        :param epoch_id: the epoch that need to be visualized
        :param data: torch.Tensor
        :param targets: numpy.ndarray
        :return: result, ndarray of boolean
        '''
        model_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "subject_model.pth")
        # try:
        #     self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        # except FileNotFoundError:
        #     print("subject model does not exist!")
        #     return None
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        fc_model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))
        data = data.to(device=self.device, dtype=torch.float)
        pred = batch_run(fc_model, data, self.class_num)
        pred = pred.argmax(axis=1)
        result = (pred == targets)
        return result

    def get_representation_data(self, epoch_id, data):
        '''
        get representation data from original image
        :param data: torch.Tensor
        :param epoch_id: the number of epoch
        :return: representation data, numpy.ndarray
        '''
        model_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "subject_model.pth")
        # try:
        #     self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        # except FileNotFoundError:
        #     print("subject model does not exist!")
        #     return None
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        repr_model = torch.nn.Sequential(*(list(self.model.children())[:self.split]))

        data = data.to(self.device)
        representation_data = batch_run(repr_model, data, self.repr_num)
        return representation_data

    def get_data_pool_repr(self, epoch_id):
        """get representations of data pool"""
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "train_data.npy")
        if os.path.exists(location):
            train_data = np.load(location)
            return train_data
        else:
            print("No data!")
            return None

    def get_pred(self, epoch_id, data):
        '''
        get the prediction score for data in epoch_id
        :param data: numpy.ndarray
        :param epoch_id:
        :return: pred, numpy.ndarray
        '''
        model_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "subject_model.pth")
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        fc_model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))

        data = torch.from_numpy(data)
        data = data.to(self.device)
        pred = batch_run(fc_model, data, self.class_num)
        return pred

    def get_epoch_train_pred(self, epoch_id):
        '''
        get the prediction score for data in epoch_id
        :param data: numpy.ndarray
        :param epoch_id:
        :return: pred, numpy.ndarray
        '''
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "train_pred_labels.npy")
        if os.path.exists(location):
            pred = np.load(location)
            return pred
        model_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "subject_model.pth")
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        fc_model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))

        data = self.get_epoch_train_repr_data(epoch_id)
        data = torch.from_numpy(data)
        data = data.to(self.device)
        pred = batch_run(fc_model, data, self.class_num)
        np.save(location, pred)
        return pred

    def get_epoch_test_pred(self, epoch_id):
        '''
        get the prediction score for data in epoch_id
        :param data: numpy.ndarray
        :param epoch_id:
        :return: pred, numpy.ndarray
        '''
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "test_pred_labels.npy")
        if os.path.exists(location):
            pred = np.load(location)
            return pred
        model_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "subject_model.pth")
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        fc_model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))

        data = self.get_epoch_test_repr_data(epoch_id)
        data = torch.from_numpy(data)
        data = data.to(self.device)
        pred = batch_run(fc_model, data, self.class_num)
        np.save(location, pred)
        return pred

    def get_epoch_train_repr_data(self, epoch_id):
        """get representations of training data"""
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "train_data.npy")
        if os.path.exists(location):
            train_data = np.load(location)
            current_index = self.get_epoch_index(epoch_id)
            train_data = train_data[current_index]
            return train_data
        else:
            print("No data!")
            return None

    def get_epoch_train_labels(self, epoch_id):
        """get training labels"""
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        if os.path.exists(index_file):
            index = load_labelled_data_index(index_file)
            labels = self.training_labels[index].cpu().numpy()
            return labels
        else:
            print("No data!")
            return None

    def get_epoch_test_repr_data(self, epoch_id):
        """get representations of testing data"""
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "test_data.npy")
        if os.path.exists(location):
            test_data = np.load(location)
            return test_data
        else:
            print("No data!")
            return None

    def get_epoch_test_labels(self, epoch_id=None):
        """get representations of testing data"""
        labels = self.testing_labels.cpu().numpy()
        if epoch_id is not None:
            test_index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
            if os.path.exists(test_index_file):
                index = load_labelled_data_index(test_index_file)
                return labels[index]
        return labels

    def batch_embedding(self, data, epoch_id):
        '''
        get embedding of subject model at epoch_id
        :param data: torch.Tensor
        :param epoch_id:
        :return: embedding, numpy.array
        '''
        repr_data = self.get_representation_data(epoch_id, data)
        embedding = self.batch_project(repr_data, epoch_id)
        return embedding

    def get_epoch_border_centers(self, epoch_id):
        """get border representations"""
        if self.advance_border_gen:
            location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "advance_border_centers.npy")
        else:
            location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "border_centers.npy")
        if os.path.exists(location):
            data = np.load(location)
            return data
        else:
            print("No data!")
            return None

    ################################################## Visualization ##################################################
    def get_epoch_plot_measures(self, epoch_id):
        """get plot measure for visualization"""
        train_data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)

        embedded = encoder(train_data).cpu().numpy()

        ebd_min = np.min(embedded, axis=0)
        ebd_max = np.max(embedded, axis=0)
        ebd_extent = ebd_max - ebd_min

        x_min, y_min = ebd_min - 0.1 * ebd_extent
        x_max, y_max = ebd_max + 0.1 * ebd_extent

        x_min = min(x_min, y_min)
        y_min = min(x_min, y_min)
        x_max = max(x_max, y_max)
        y_max = max(x_max, y_max)

        return x_min, y_min, x_max, y_max

    def get_epoch_decision_view(self, epoch_id, resolution=-1):
        '''
        get background classifier view
        :param epoch_id: epoch that need to be visualized
        :param resolution: background resolution
        :return:
            grid_view : numpy.ndarray, self.resolution,self.resolution, 2
            decision_view : numpy.ndarray, self.resolution,self.resolution, 3
        '''
        if self.verbose > 0:
            print('Computing decision regions ...')
        if resolution == -1:
            resolution = self.resolution

        decoder = self.get_inv_model(epoch_id)

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch_id)

        # create grid
        xs = np.linspace(x_min, x_max, resolution)
        ys = np.linspace(y_min, y_max, resolution)
        grid = np.array(np.meshgrid(xs, ys))
        grid = np.swapaxes(grid.reshape(grid.shape[0], -1), 0, 1)

        # map gridmpoint to images
        grid_samples = decoder(grid).cpu().numpy()
        mesh_preds = self.get_pred(epoch_id, grid_samples)
        mesh_preds = mesh_preds + 1e-8

        sort_preds = np.sort(mesh_preds)
        diff = (sort_preds[:, -1] - sort_preds[:, -2]) / (sort_preds[:, -1] - sort_preds[:, 0])
        border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        border[diff < 0.15] = 1
        diff[border == 1] = 0.

        diff = diff/diff.max()
        diff = diff*0.9

        mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(mesh_classes)
        color = self.cmap(mesh_classes / mesh_max_class)

        diff = diff.reshape(-1, 1)

        color = color[:, 0:3]
        # color = diff * 0.5 * color + (1 - diff) * np.ones(color.shape, dtype=np.uint8)
        color = diff * 0.5 * color + (1 - diff) * np.ones(color.shape, dtype=np.uint8)
        decision_view = color.reshape(resolution, resolution, 3)
        grid_view = grid.reshape(resolution, resolution, 2)
        return grid_view, decision_view

    def get_standard_classes_color(self):
        '''
        get the RGB value for 10 classes
        :return:
            color : numpy.ndarray, shape (10, 3)
        '''
        mesh_max_class = self.class_num - 1
        mesh_classes = np.arange(10)
        color = self.cmap(mesh_classes / mesh_max_class)

        color = color[:, 0:3]
        return color

    def _init_plot(self, is_for_frontend=False):
        '''
        Initialises matplotlib artists and plots. from DeepView
        '''
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))
        if not is_for_frontend:
            # self.ax.set_title(self.title)
            self.ax.set_title("DVI visualization")
        self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        # labels = prediction
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=1,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        # labels != prediction, labels be a large circle
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                fillstyle='full', ms=3, mew=2.5, zorder=3)
            self.sample_plots.append(plot[0])

        # labels != prediction, prediction stays inside of circle
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], '.', markeredgecolor=color,
                                fillstyle='full', ms=2, zorder=4)
            self.sample_plots.append(plot[0])

        # highlight
        color = (0.0, 0.0, 0.0, 1.0)
        plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                            fillstyle='full', ms=4, mew=4, zorder=1)
        self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        # self.fig.canvas.mpl_connect('pick_event', self.show_sample)
        # self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False
        if not is_for_frontend:
            self.ax.legend()

    def show(self, epoch_id):
        '''
        Shows the current plot. from DeepView
        '''
        # if not hasattr(self, 'fig'):
        #     self._s()
        self._init_plot()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch_id)

        grid, decision_view = self.get_epoch_decision_view(epoch_id, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'res: %d'
        desc = params_str % (self.resolution)
        self.desc.set_text(desc)

        train_data = self.get_epoch_train_repr_data(epoch_id)
        train_labels = self.get_epoch_train_labels(epoch_id)
        pred = self.get_pred(epoch_id, train_data)
        pred = pred.argmax(axis=1)

        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(train_data).cpu().numpy()

        # labels == pred
        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        # labels != pred, draw label
        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels != pred)]
            self.sample_plots[self.class_num + c].set_data(data.transpose())
        # labels != pred, draw pred
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, train_labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show()

    def savefig(self, epoch_id, path):
        '''
        Shows the current plot.
        '''
        # if not hasattr(self, 'fig'):
        #     self._s()
        self._init_plot()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch_id)

        grid, decision_view = self.get_epoch_decision_view(epoch_id, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        params_str = 'res: %d'
        desc = params_str % (self.resolution)
        self.desc.set_text(desc)

        train_data = self.get_epoch_train_repr_data(epoch_id)
        train_labels = self.get_epoch_train_labels(epoch_id)
        pred = self.get_pred(epoch_id, train_data)
        pred = pred.argmax(axis=1)

        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(train_data).cpu().numpy()
        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(train_labels == c, train_labels != pred)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())
        #
        for c in range(self.class_num):
            data = embedding[np.logical_and(pred == c, train_labels != pred)]
            self.sample_plots[2*self.class_num + c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path)

    def _customized_init(self):
        '''
        Initialises matplotlib artists and plots. from DeepView
        '''
        # if self.interactive:
        #     plt.ion()
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self.ax.set_axis_off()

        # self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
            interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        # train_data labels
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=2,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        # testing wrong labels
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '^', markeredgecolor=color,
                fillstyle='full', ms=3, mew=3, zorder=1)
            self.sample_plots.append(plot[0])

        # highlight
        color = (0.0, 0.0, 0.0, 1.0)
        plot = self.ax.plot([], [], '.', markeredgecolor=color,
                            fillstyle='full', ms=0.5, mew=0.5, zorder=3)
        self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        # self.fig.canvas.mpl_connect('pick_event', self.show_sample)
        # self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False

    def customize_visualize(self, epoch_id, test_data, test_labels, path):
        '''
        Shows the current plot.
        training data in small dot, test data in larger triangle, highlight in red
        '''
        # if not hasattr(self, 'fig'):
        #     self._s()
        self._customized_init()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch_id)

        grid, decision_view = self.get_epoch_decision_view(epoch_id, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))


        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(test_data).cpu().numpy()
        # test_embedding = proj_encoder(test_data).cpu().numpy()
        for c in range(self.class_num):
            # data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
            data = embedding[(test_labels == c)]
            self.sample_plots[c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    def _al_visualize_init(self):
        '''
        Initialises matplotlib artists and plots. from DeepView
        '''
        # if self.interactive:
        #     plt.ion()
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 8))

        self.ax.set_axis_off()

        # self.desc = self.fig.text(0.5, 0.02, '', fontsize=8, ha='center')
        self.cls_plot = self.ax.imshow(np.zeros([5, 5, 3]),
                                       interpolation='gaussian', zorder=0, vmin=0, vmax=1)

        self.sample_plots = []

        # train_data labels
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], 'o', label=self.classes[c], ms=1.5,
                                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        # testing wrong labels
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], '^', markeredgecolor=color,
                                fillstyle='full', ms=3, mew=3, zorder=1)
            self.sample_plots.append(plot[0])

        # highlight with labels
        for c in range(self.class_num):
            color = self.cmap(c / (self.class_num - 1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                                fillstyle='full', ms=2.5, mew=2.5, zorder=4)
            self.sample_plots.append(plot[0])
        # highlight
        color = (0.0, 0.0, 0.0, 1.0)
        plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                            fillstyle='full', ms=2.8, mew=2.8, zorder=3)
        self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        # self.fig.canvas.mpl_connect('pick_event', self.show_sample)
        # self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False

    def al_visualize(self, epoch_id, train_data, train_labels, path, highlight_index):
        '''
        active learning visualization implementation
        '''
        self._al_visualize_init()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch_id)

        grid, decision_view = self.get_epoch_decision_view(epoch_id, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(train_data).cpu().numpy()

        # highlight in label color
        highlight_embedding = embedding[highlight_index]
        highlight_labels = train_labels[highlight_index]
        for c in range(self.class_num):
            data = highlight_embedding[highlight_labels == c]
            self.sample_plots[2 * self.class_num + c].set_data(data.transpose())
        # highlight in black edge
        data = embedding[highlight_index]
        self.sample_plots[3 * self.class_num].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    ############################################## Evaluation Functions ###############################################

    '''projection preserving'''
    def proj_nn_perseverance_knn_train(self, epoch_id, n_neighbors=15):
        """evalute training nn preserving property"""
        train_data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(train_data).cpu().numpy()

        del encoder
        gc.collect()

        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors, metric="euclidean")
        return val

    def proj_nn_perseverance_knn_test(self, epoch_id, n_neighbors=15):
        """evalute testing nn preserving property"""
        # test_data = self.get_representation_data(epoch_id, self.testing_data)
        test_data = self.get_epoch_test_repr_data(epoch_id)
        train_data = self.get_epoch_train_repr_data(epoch_id)

        fitting_data = np.concatenate((train_data, test_data), axis=0)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(fitting_data).cpu().numpy()

        del encoder
        gc.collect()

        val = evaluate_proj_nn_perseverance_knn(fitting_data, embedding, n_neighbors, metric="euclidean")
        return val

    def proj_boundary_perseverance_knn_train(self, epoch_id, n_neighbors=15):
        """evalute training boundary preserving property"""
        encoder = self.get_proj_model(epoch_id)
        border_centers = self.get_epoch_border_centers(epoch_id)
        train_data = self.get_epoch_train_repr_data(epoch_id)

        low_center = encoder(border_centers).cpu().numpy()
        low_train = encoder(train_data).cpu().numpy()

        del encoder
        gc.collect()

        val = evaluate_proj_boundary_perseverance_knn(train_data, low_train, border_centers, low_center, n_neighbors)

        return val

    def proj_boundary_perseverance_knn_test(self, epoch_id, n_neighbors=15):
        """evalute testing boundary preserving property"""
        encoder = self.get_proj_model(epoch_id)
        border_centers = self.get_epoch_border_centers(epoch_id)
        train_data = self.get_epoch_train_repr_data(epoch_id)
        test_data = self.get_epoch_test_repr_data(epoch_id)
        fitting_data = np.concatenate((train_data, test_data), axis=0)

        low_center = encoder(border_centers).cpu().numpy()
        low_data = encoder(fitting_data).cpu().numpy()

        del encoder
        gc.collect()

        val = evaluate_proj_boundary_perseverance_knn(fitting_data, low_data, border_centers, low_center, n_neighbors)

        return val

    def proj_temporal_perseverance_train(self, n_neighbors=15, eval_name=""):
        """evalute training temporal preserving property"""
        l = load_labelled_data_index(os.path.join(self.model_path, "Epoch_{:d}".format(self.epoch_start), "index.json"))
        l = len(l)
        eval_num = int((self.epoch_end - self.epoch_start) / self.period)
        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))
        for n_epoch in range(self.epoch_start+self.period, self.epoch_end+1, self.period):

            prev_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch - self.period), "train_data.npy")
            prev_index = self.get_epoch_index(n_epoch - self.period)
            prev_data = np.load(prev_data_loc)[prev_index]

            encoder = self.get_proj_model(n_epoch - self.period)
            prev_embedding = encoder(prev_data).cpu().numpy()

            del encoder
            gc.collect()

            encoder = self.get_proj_model(n_epoch)
            data = self.get_epoch_train_repr_data(n_epoch)
            embedding = encoder(data).cpu().numpy()

            del encoder
            gc.collect()

            alpha_ = backend.find_neighbor_preserving_rate(prev_data, data, n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - embedding, axis=1)

            alpha[int((n_epoch - self.epoch_start) / self.period - 1)] = alpha_
            delta_x[int((n_epoch - self.epoch_start) / self.period - 1)] = delta_x_

        # val_entropy = evaluate_proj_temporal_perseverance_entropy(alpha, delta_x)
        val_corr = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
        # save result
        save_dir = os.path.join(self.model_path,  "time{}.json".format(eval_name))
        if not os.path.exists(save_dir):
            evaluation = dict()
        else:
            f = open(save_dir, "r")
            evaluation = json.load(f)
            f.close()
        evaluation["temporal_train_{}".format(n_neighbors)] = val_corr
        with open(save_dir, "w") as f:
            json.dump(evaluation, f)
        if self.verbose:
            print("succefully save train corr {}".format(val_corr))
        return val_corr

    def proj_temporal_perseverance_test(self, n_neighbors=15, eval_name=""):
        """evalute testing temporal preserving property"""
        l = os.path.join(self.model_path, "Epoch_{:d}".format(self.epoch_start), "test_index.json")
        if os.path.exists(l):
            l = load_labelled_data_index(l)
            l = len(l)
        else:
            l = len(self.testing_labels)
        eval_num = int((self.epoch_end - self.epoch_start) / self.period)
        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))
        for n_epoch in range(self.epoch_start + self.period, self.epoch_end + 1, self.period):

            # prev_data = self.get_representation_data(n_epoch-self.period, self.testing_data)
            prev_data = self.get_epoch_test_repr_data(n_epoch - self.period)
            encoder = self.get_proj_model(n_epoch - self.period)
            prev_embedding = encoder(prev_data).cpu().numpy()

            del encoder
            gc.collect()

            # data = self.get_representation_data(n_epoch, self.testing_data)
            encoder = self.get_proj_model(n_epoch)
            data = self.get_epoch_test_repr_data(n_epoch)
            embedding = encoder(data).cpu().numpy()

            del encoder
            gc.collect()

            alpha_ = backend.find_neighbor_preserving_rate(prev_data, data, n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - embedding, axis=1)
            alpha[int((n_epoch - self.epoch_start) / self.period - 1)] = alpha_
            delta_x[int((n_epoch - self.epoch_start) / self.period - 1)] = delta_x_

        val_corr = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)

        # save result
        save_dir = os.path.join(self.model_path,  "time{}.json".format(eval_name))
        if not os.path.exists(save_dir):
            evaluation = dict()
        else:
            f = open(save_dir, "r")
            evaluation = json.load(f)
            f.close()
        evaluation["temporal_test_{}".format(n_neighbors)] = val_corr
        with open(save_dir, "w") as f:
            json.dump(evaluation, f)
        if self.verbose:
            print("successfully save test corr {}".format(val_corr))

        return val_corr

    '''inverse preserving'''
    def inv_accu_train(self, epoch_id):
        """inverse training prediction accuracy"""
        data = self.get_epoch_train_repr_data(epoch_id)

        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        pred = self.get_pred(epoch_id, data).argmax(axis=1)
        new_pred = self.get_pred(epoch_id, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
        return val

    def inv_accu_test(self, epoch_id):
        """inverse testing prediction accuracy"""
        # data = self.get_representation_data(epoch_id, self.testing_data)
        data = self.get_epoch_test_repr_data(epoch_id)

        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        pred = self.get_pred(epoch_id, data).argmax(axis=1)
        new_pred = self.get_pred(epoch_id, inv_data).argmax(axis=1)

        val = evaluate_inv_accu(pred, new_pred)
        return val

    def inv_dist_train(self, epoch_id):
        """inverse training difference"""
        data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        val = evaluate_inv_distance(data, inv_data)
        return val

    def inv_dist_test(self, epoch_id):
        """inverse testing difference"""
        # data = self.get_representation_data(epoch_id, self.testing_data)
        data = self.get_epoch_test_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        val = evaluate_inv_distance(data, inv_data)
        return val

    def inv_conf_diff_train(self, epoch_id):
        """inverser training confidence difference"""
        data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        ori_pred = self.get_pred(epoch_id, data)
        new_pred = self.get_pred(epoch_id, inv_data)
        labels = ori_pred.argmax(-1)

        ori_pred = softmax(ori_pred, axis=1)
        new_pred = softmax(new_pred, axis=1)

        val = evaluate_inv_conf(labels, ori_pred, new_pred)
        return val

    def inv_conf_diff_test(self, epoch_id):
        """inverse testing confidence difference"""
        # data = self.get_representation_data(epoch_id, self.testing_data)
        data = self.get_epoch_test_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        ori_pred = self.get_pred(epoch_id, data)
        new_pred = self.get_pred(epoch_id, inv_data)
        labels = ori_pred.argmax(-1)

        ori_pred = softmax(ori_pred, axis=1)
        new_pred = softmax(new_pred, axis=1)

        val = evaluate_inv_conf(labels, ori_pred, new_pred)
        return val

    def keep_B_train(self, epoch_id, resolution=400, threshold=0.7):
        train_data = self.get_epoch_train_repr_data(epoch_id)
        preds = self.get_pred(epoch_id, train_data)
        is_border = is_B(preds)
        border_points = train_data[is_border]
        grid_view, decision_view = self.get_epoch_decision_view(epoch_id, resolution=resolution)
        low_B = self.batch_project(border_points, epoch_id)

        ans = evaluate_keep_B(low_B, grid_view, decision_view, threshold=threshold)
        if self.verbose:
            print("{:.2f}% of training boundary points still lie on boundary after dimension reduction...".format(ans*100))
        return ans

    def keep_B_test(self, epoch_id, resolution=400, threshold=0.7):
        test_data = self.get_epoch_test_repr_data(epoch_id)
        preds = self.get_pred(epoch_id, test_data)
        is_border = is_B(preds)
        border_points = test_data[is_border]

        grid_view, decision_view = self.get_epoch_decision_view(epoch_id, resolution=resolution)
        low_B = self.batch_project(border_points, epoch_id)

        ans = evaluate_keep_B(low_B, grid_view, decision_view, threshold=threshold)
        if self.verbose:
            print("{:.2f}% of testing boundary points still lie on boundary after dimension reduction...".format(ans*100))
        return ans

    def keep_B_boundary(self, epoch_id, resolution=400, threshold=0.7):
        border_points = self.get_epoch_border_centers(epoch_id)
        grid_view, decision_view = self.get_epoch_decision_view(epoch_id, resolution=resolution)
        low_B = self.batch_project(border_points, epoch_id)

        ans = evaluate_keep_B(low_B, grid_view, decision_view, threshold=threshold)
        if self.verbose:
            print("{:.2f}% of boundary points still lie on boundary after dimension reduction...".format(ans*100))
        return ans

    def point_inv_preserve(self, epoch_id, data):
        """
        get inverse confidence for a single point
        :param epoch_id: int
        :param data: numpy.ndarray
        :return l: boolean, whether reconstruction data have the same prediction
        :return conf_diff: float, (0, 1), confidence difference
        """
        data = np.expand_dims(data, 0)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        ori_pred = self.get_pred(epoch_id, data).squeeze()
        new_pred = self.get_pred(epoch_id, inv_data).squeeze()
        old_label = ori_pred.argmax(-1)
        new_label = new_pred.argmax(-1)
        l = old_label == new_label

        conf_diff = np.abs(ori_pred[old_label] - new_pred[old_label])

        return l, conf_diff

    def batch_inv_preserve(self, epoch_id, data):
        """
        get inverse confidence for a single point
        :param epoch_id: int
        :param data: numpy.ndarray
        :return l: boolean, whether reconstruction data have the same prediction
        :return conf_diff: float, (0, 1), confidence difference
        """
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        ori_pred = self.get_pred(epoch_id, data)
        new_pred = self.get_pred(epoch_id, inv_data)
        ori_pred = softmax(ori_pred, axis=1)
        new_pred = softmax(new_pred, axis=1)

        old_label = ori_pred.argmax(-1)
        new_label = new_pred.argmax(-1)
        l = old_label == new_label

        old_conf = [ori_pred[i, old_label[i]] for i in range(len(old_label))]
        new_conf = [new_pred[i, old_label[i]] for i in range(len(old_label))]
        old_conf = np.array(old_conf)
        new_conf = np.array(new_conf)

        conf_diff = old_conf - new_conf

        return l, conf_diff

    def get_eval(self, epoch_id):
        with open(os.path.join(self.model_path, "Epoch_{}".format(epoch_id),"evaluation.json"), 'r') as f:
            evaluation = json.load(f)
            evaluation_new = evaluation
            for item in evaluation:
                value = evaluation[item]
                value = round(value, 2)
                evaluation_new[item] = value
        return evaluation_new


    '''subject model'''
    def training_accu(self, epoch_id):
        train_data = self.get_epoch_train_repr_data(epoch_id)
        labels = self.get_epoch_train_labels(epoch_id)
        pred = self.get_pred(epoch_id, train_data).argmax(-1)
        val = evaluate_inv_accu(labels, pred)
        return val

    def testing_accu(self, epoch_id):
        # test_data = self.testing_data
        labels = self.testing_labels.cpu().numpy()
        test_index_file = os.path.join(self.model_path, "Epoch_{}".format(epoch_id), "test_index.json")
        if os.path.exists(test_index_file):
            index = load_labelled_data_index(test_index_file)
            labels = labels[index]
        # repr_data = self.get_representation_data(epoch_id, test_data)
        repr_data = self.get_epoch_test_repr_data(epoch_id)
        pred = self.get_pred(epoch_id, repr_data).argmax(-1)
        val = evaluate_inv_accu(labels, pred)
        return val

    def get_dataset_length(self):
        return len(self.training_labels) + len(self.testing_labels)

    ############################################## Case Studies Related ###############################################
    '''active learning'''
    def get_new_index(self, epoch_id):
        """get the index of new selection"""
        new_epoch = epoch_id + self.period
        if new_epoch > self.epoch_end:
            return list()

        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        index = load_labelled_data_index(index_file)
        new_index_file = os.path.join(self.model_path, "Epoch_{:d}".format(new_epoch), "index.json")
        new_index = load_labelled_data_index(new_index_file)

        idxs = []
        for i in new_index:
            if i not in index:
                idxs.append(i)

        return idxs

    def get_epoch_index(self, epoch_id):
        """get the training data index for an epoch"""
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        index = load_labelled_data_index(index_file)
        return index

    '''Noise data(Mislabelled data)'''
    def noisy_data_index(self):
        """get noise data index"""
        index_file = os.path.join(self.content_path, "index.json")
        if not os.path.exists(index_file):
            return list()
        return load_labelled_data_index(index_file)

    def get_original_labels(self):
        """
        get original dataset label(without noise)
        :return labels: list, shape(N,)
        """
        index_file = os.path.join(self.content_path, "index.json")
        if not os.path.exists(index_file):
            return list()
        index = load_labelled_data_index(index_file)
        old_file = os.path.join(self.content_path, "old_labels.json")
        old_labels = load_labelled_data_index(old_file)

        labels = np.copy(self.training_labels.cpu().numpy())
        labels[index] = old_labels

        return labels

    def get_uncertainty_score(self, epoch_id):
        try:
            uncertainty_score_path = os.path.join(self.model_path, "Epoch_{}".format(epoch_id),"train_uncertainty_score.json")
            with open(uncertainty_score_path, "r") as f:
                train_uncertainty_score = json.load(f)

            uncertainty_score_path = os.path.join(self.model_path, "Epoch_{}".format(epoch_id),"test_uncertainty_score.json")
            with open(uncertainty_score_path, "r") as f:
                test_uncertainty_score = json.load(f)

            uncertainty_score = train_uncertainty_score + test_uncertainty_score
            return uncertainty_score
        except FileNotFoundError:
            train_num = self.training_labels.shape[0]
            test_num = self.testing_labels.shape[0]
            return [-1 for i in range(train_num+test_num)]

    def get_diversity_score(self, epoch_id):
        try:
            dis_score_path = os.path.join(self.model_path, "Epoch_{}".format(epoch_id), "train_dis_score.json")
            with open(dis_score_path, "r") as f:
                train_dis_score = json.load(f)

            dis_score_path = os.path.join(self.model_path, "Epoch_{}".format(epoch_id), "test_dis_score.json")
            with open(dis_score_path, "r") as f:
                test_dis_score = json.load(f)

            dis_score = train_dis_score + test_dis_score

            return dis_score
        except FileNotFoundError:
            train_num = self.training_labels.shape[0]
            test_num = self.testing_labels.shape[0]
            return [-1 for i in range(train_num+test_num)]

    def get_total_score(self, epoch_id):
        try:
            total_score_path = os.path.join(self.model_path, "Epoch_{}".format(epoch_id), "train_total_score.json")
            with open(total_score_path, "r") as f:
                train_total_score = json.load(f)

            total_score_path = os.path.join(self.model_path, "Epoch_{}".format(epoch_id), "test_total_score.json")
            with open(total_score_path, "r") as f:
                test_total_score = json.load(f)

            total_score = train_total_score + test_total_score

            return total_score
        except FileNotFoundError:
            train_num = self.training_labels.shape[0]
            test_num = self.testing_labels.shape[0]
            return [-1 for i in range(train_num+test_num)]

    def save_DVI_selection(self, epoch_id, indices):
        """
        save the selected index message from DVI frontend
        :param epoch_id:
        :param indices: list, selected indices
        :return:
        """
        save_location = os.path.join(self.model_path, "Epoch_{}".format(epoch_id), "DVISelection.json")
        with open(save_location, "w") as f:
            json.dump(indices, f)
    ######################################## DVI tensorboard frontend ###############################################

    # TODO setup APIs to get attributes for DVI frontend
    # Subject Model table
    def subject_model_table(self):
        """
        get the dataframe for subject model table
        :return:
        """
        path_list = []
        epoch_list = []
        train_accu = []
        test_accu = []
        for n_epoch in range(self.epoch_start, self.epoch_end+1, self.period):
            path = os.path.join(self.model_path, "Epoch_{}".format(n_epoch), "subject_model.pth")
            path_list.append(path)
            epoch_list.append(n_epoch)
            train_accu.append(self.training_accu(n_epoch))
            test_accu.append(self.testing_accu(n_epoch))
        df_dict = {
            "location": path_list,
            "epoch": epoch_list,
            "train_accu": train_accu,
            "test_accu": test_accu
        }
        df = pd.DataFrame(df_dict, index=pd.Index(range(len(path_list)), name="idx"))
        return df

    # Visualization model table
    def vis_model_table(self):
        """
        get the dataframe for vis model table
        :return:
        """
        temporal = False
        if self.temporal:
            temporal = True
        path_list = []
        epoch_list = []
        temporal_list = []
        nn_train = []
        boundary_train = []
        ppr_train = []
        ccr_train = []
        nn_test = []
        boundary_test = []
        ppr_test = []
        ccr_test = []
        for n_epoch in range(self.epoch_start, self.epoch_end+1, self.period):
            path = os.path.join(self.model_path, "Epoch_{}".format(n_epoch))
            path_list.append(path)
            epoch_list.append(n_epoch)
            temporal_list.append(temporal)

            eval_path = os.path.join(self.model_path, "Epoch_{}".format(n_epoch), "evaluation.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train.append(eval["nn_train_15"])
            nn_test.append(eval["nn_test_15"])
            boundary_train.append(eval["bound_train_15"])
            boundary_test.append(eval["bound_test_15"])
            ppr_train.append(eval["inv_acc_train"])
            ppr_test.append(eval["inv_acc_test"])
            ccr_train.append(eval["inv_conf_train"])
            ccr_test.append(eval["inv_conf_test"])
        df_dict = {
            "location": path_list,
            "epoch": epoch_list,
            "temporal_loss": temporal_list,
            "nn_train": nn_train,
            "nn_test": nn_test,
            "boundary_train": boundary_train,
            "boundary_test": boundary_test,
            "ppr_train": ppr_train,
            "ppr_test": ppr_test,
            "ccr_train":ccr_train,
            "ccr_test": ccr_test
        }
        df = pd.DataFrame(df_dict, index=pd.Index(range(len(path_list)), name="idx"))
        # TODO check epoch_start and epoch_end
        return df

    # Sample table
    def sample_table(self):
        """
        sample table:
            label
            index
            type:["train", "test", others...]
            customized attributes
        :return:
        """
        train_labels = self.training_labels.cpu().numpy().tolist()
        test_labels = self.testing_labels.cpu().numpy().tolist()
        labels = train_labels + test_labels
        train_type = ["train" for i in range(len(train_labels))]
        test_type = ["test" for i in range(len(test_labels))]
        types = train_type + test_type
        df_dict = {
            "labels": labels,
            "type": types
        }

        df = pd.DataFrame(df_dict, index=pd.Index(range(len(labels)), name="idx"))
        return df

    def sample_table_AL(self):
        """
        sample table for active learning scenarios
        :return:
        """
        df = self.sample_table()
        new_selected_epoch = [-1 for _ in range(len(self.training_labels)+len(self.testing_labels))]
        new_selected_epoch = np.array(new_selected_epoch)
        for epoch_id in range(self.epoch_start, self.epoch_end+1, self.period):
            labeled = np.array(self.get_epoch_index(epoch_id))
            new_selected_epoch[labeled] = epoch_id
        df["al_selected_epoch"] = new_selected_epoch.tolist()
        return df

    def sample_table_noisy(self):
        df = self.sample_table()
        noisy_data = self.noisy_data_index()
        is_noisy = np.array([False for _ in range(len(self.training_labels)+len(self.testing_labels))])
        is_noisy[noisy_data] = True

        original_label = self.get_original_labels().tolist()
        test_labels = self.testing_labels.cpu().numpy().tolist()
        original_label = original_label.extend(test_labels)

        df["original_label"] = original_label
        df["is_noisy"] = is_noisy.tolist()
        return df

    # customized features
    def filter_label(self, label):
        try:
            index = self.classes.index(label)
        except:
            index = -1
        train_labels = self.training_labels.cpu().numpy()
        test_labels = self.testing_labels.cpu().numpy()
        labels = np.concatenate((train_labels, test_labels), 0)
        idxs = np.argwhere(labels == index)
        idxs = np.squeeze(idxs)
        return idxs

    def filter_type(self, type, epoch_id):
        if type == "train":
            res = self.get_epoch_index(epoch_id)
        elif type == "test":
            train_num = self.training_labels.cpu().numpy().shape[0]
            test_num = self.testing_labels.cpu().numpy().shape[0]
            res = list(range(train_num, test_num, 1))
        elif type == "unlabel":
            labeled = np.array(self.get_epoch_index(epoch_id))
            train_num = self.training_labels.cpu().numpy().shape[0]
            all_data = np.arange(train_num)
            unlabeled = np.setdiff1d(all_data, labeled)
            res = unlabeled.tolist()
        elif type == "noisy":
            res = self.noisy_data_index()
            print(res)
        else:
            # all data
            train_num = self.training_labels.cpu().numpy().shape[0]
            test_num = self.testing_labels.cpu().numpy().shape[0]
            res = list(range(0, train_num + test_num, 1))
        return res

    def filter_prediction(self, pred):
        pass

