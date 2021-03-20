import tensorflow as tf
from deepvisualinsight.utils import *
from deepvisualinsight.backend import *
import matplotlib.pyplot as plt
import matplotlib as mpl
from deepvisualinsight.evaluate import *
import gc
from scipy.special import softmax
from scipy.spatial.distance import cdist
import deepvisualinsight.utils_advanced as utils_advanced



class MMS:
    def __init__(self, content_path, model_structure, epoch_start, epoch_end, period, repr_num, class_num, classes, low_dims=2,
                 cmap="tab10", resolution=100, neurons=None, temporal=False, transfer_learning=True, verbose=1,
                 split=-1, advance_border_gen=False, attack_device="cpu"):
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
            the output shape of representation data
        class_num : int
            the length of classification labels
        classes	: list, tuple of str
            All classes that the classifier uses as a list/tuple of strings.
        low_dims: tuple
            the expected low dimension shape
        cmap : str, by default 'tab10'
            Name of the colormap to use for visualization.
            The number of distinguishable colors should correspond to class_num.
            See here for the names: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
        resolution : int, by default 100
            Resolution of the classification boundary plot.
        # boundary_diff : float
        #     the difference between top1 and top2 logits from last layer. The difference is used to define boundary.
        neurons : int
            the number of units inside each layer of autoencoder
        temporal: boolean, by default False
            choose whether to add temporal loss or not
        transfer_learning: boolean, by default True
            choose whether to use transfer learning
        verbose : int, by default 1
        split: int, by default -1
        advance_border_gen : boolean, by default False
            whether to use advance adversarial attack method for border points generation
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
        # self.boundary_diff = boundary_diff
        self.class_num = class_num
        self.classes = classes
        self.temporal = temporal
        self.transfer_learning = transfer_learning
        self.verbose = verbose
        self.split = split
        self.advance_border_gen = advance_border_gen
        self.device = torch.device(attack_device)
        # self.device = torch.device('cpu')
        if len(tf.config.list_physical_devices('GPU')) > 0:
            self.tf_device = tf.config.list_physical_devices('GPU')[0]
            for d in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(d, True)
        else:
            self.tf_device = tf.config.list_physical_devices('CPU')[0]
        if neurons is None:
            self.neurons = self.repr_num / 2
        else:
            self.neurons = neurons
        self.load_content()

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

    def data_preprocessing(self):
        '''
        preprocessing data. This process includes find_train_centers, find_border_points and find_border_centers
        save data for later training
        '''
        time_border_clustering = list()
        time_train_clustering = list()
        time_borders_gen = list()
        for n_epoch in range(self.epoch_start, self.epoch_end+1, self.period):
            index_file = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "index.json")
            index = load_labelled_data_index(index_file)
            training_data = self.training_data[index]
            testing_data = self.testing_data
            training_labels = self.training_labels[index]

            model_location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "subject_model.pth")
            self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
            self.model = self.model.to(self.device)
            self.model.eval()

            repr_model = torch.nn.Sequential(*(list(self.model.children())[:self.split]))

            n_clusters = math.floor(len(index) / 10)

            if self.advance_border_gen:
                t0 = time.time()
                gaps, preds, confs = utils_advanced.batch_run(self.model, training_data, self.device, batch_size=200)
                # Kmeans clustering
                kmeans_result, predictions = utils_advanced.clustering(gaps.numpy(), preds.numpy(),
                                                                       n_clusters_per_cls=10)
                # Adversarial attacks
                border_points, _ = utils_advanced.get_border_points(model=self.model,
                                                                    input_x=training_data, gaps=gaps,
                                                                    confs=confs,
                                                                    kmeans_result=kmeans_result,
                                                                    predictions=predictions, device=self.device,
                                                                    num_adv_eg=n_clusters, num_cls=10,
                                                                    n_clusters_per_cls=10, verbose=0)
                t1 = time.time()
                time_borders_gen.append(round(t1 - t0, 4))

                # get gap layer data
                border_points = border_points.to(self.device)
                border_centers = batch_run(repr_model, border_points, self.repr_num)
                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "advance_border_centers.npy")
                np.save(location, border_centers)

                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "ori_advance_border_centers.npy")
                np.save(location, border_points.cpu().numpy())
            else:
                # border points gen
                t0 = time.time()
                border_points = get_border_points(training_data, training_labels, self.model, self.device)
                t1 = time.time()
                time_borders_gen.append(round(t1 - t0, 4))

                # border clustering
                border_points = torch.from_numpy(border_points)
                border_points = border_points.to(self.device)
                border_representation = batch_run(repr_model, border_points, self.repr_num)
                t2 = time.time()
                border_centers = clustering(border_representation, n_clusters, verbose=0)
                t3 = time.time()
                time_border_clustering.append(round(t3 - t2, 4))
                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "border_centers.npy")
                np.save(location, border_centers)

                index = cdist(border_centers, border_representation, 'euclidean').argmax(-1)
                ori_border_points = border_points.cpu().numpy()[index]
                location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "ori_border_centers.npy")
                np.save(location, ori_border_points)

            # training data clustering
            train_data = training_data.to(self.device)
            train_data_representation = batch_run(repr_model, train_data, self.repr_num)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_data.npy")
            np.save(location, train_data_representation)

            t5 = time.time()
            train_centers = clustering(train_data_representation, n_clusters, verbose=0)
            t6 = time.time()
            time_train_clustering.append(round(t6 - t5, 4))
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_centers.npy")
            np.save(location, train_centers)

            # test data
            test_data = testing_data.to(self.device)
            test_data_representation = batch_run(repr_model, test_data, self.repr_num)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "test_data.npy")
            np.save(location, test_data_representation)

            if self.verbose > 0:
                print("Finish data preprocessing for Epoch {:d}...".format(n_epoch))
        time_taken = list()
        time_taken.append(sum(time_borders_gen) / len(time_borders_gen))
        time_taken.append(sum(time_train_clustering) / len(time_train_clustering))
        if len(time_border_clustering) > 0:
            time_taken.append(sum(time_border_clustering) / len(time_border_clustering))
        print(
            "Average time for generate border points: {:.4f}".format(sum(time_borders_gen) / len(time_borders_gen)))
        print("Average time for clustering training points: {:.4f}".format(
            sum(time_train_clustering) / len(time_train_clustering)))
        if len(time_border_clustering) > 0:
            print("Average time for clustering border points: {:.4f}".format(
                sum(time_border_clustering) / len(time_border_clustering)))

        self.model = self.model.to(self.device)
        return time_taken

    def prepare_visualization_for_all(self, encoder_in=None, decoder_in=None):
        """
        conduct transfer learning to save the visualization model for each epoch
        """
        dims = (self.repr_num,)
        n_components = 2
        batch_size = 200
        encoder, decoder = define_autoencoder(dims, n_components, self.neurons)
        if encoder_in is not None:
            encoder = encoder_in
        if decoder_in is not None:
            decoder = decoder_in
        optimizer = tf.keras.optimizers.Adam(1e-3)

        weights_dict = {}
        losses, loss_weights = define_losses(200, self.temporal)
        parametric_model = ParametricModel(encoder, decoder, optimizer, losses, loss_weights, self.temporal,
                                           prev_trainable_variables=None)
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
        # self.data_preprocessing()
        t0 = time.time()
        for n_epoch in range(self.epoch_start, self.epoch_end+1, self.period):
            if not self.transfer_learning:
                encoder, decoder = define_autoencoder(dims, n_components, self.neurons)
                if encoder_in is not None:
                    encoder = encoder_in
                if decoder_in is not None:
                    decoder = decoder_in
                parametric_model = ParametricModel(encoder, decoder, optimizer, losses, loss_weights,
                                                   self.temporal,
                                                   prev_trainable_variables=None)
            parametric_model.compile(
                optimizer=optimizer, loss=losses, loss_weights=loss_weights,
            )

            train_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_centers.npy")
            if self.advance_border_gen:
                border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch),
                                                  "advance_border_centers.npy")
            else:
                border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch),
                                                  "border_centers.npy")
            train_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_data.npy")

            # in case no data save for vis
            if not os.path.exists(train_data_loc):
                continue

            train_centers = np.load(train_centers_loc)
            border_centers = np.load(border_centers_loc)
            train_data = np.load(train_data_loc)

            complex, sigmas, rhos = fuzzy_complex(train_data, 15)
            bw_complex, _, _ = boundary_wise_complex(train_centers, border_centers, 15)
            if not self.temporal:
                (
                    edge_dataset,
                    _batch_size,
                    n_edges,
                    _edge_weight,
                ) = construct_mixed_edge_dataset(
                    (train_data, train_centers, border_centers),
                    complex,
                    bw_complex,
                    20,
                    batch_size,
                    parametric_embedding=True,
                    parametric_reconstruction=True,
                )
            else:

                prev_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch-self.period), "train_data.npy")
                if os.path.exists(prev_data_loc) and self.epoch_start != n_epoch:
                    prev_data = np.load(prev_data_loc)
                else:
                    prev_data = None
                if prev_data is None:
                    prev_embedding = np.zeros((len(train_data), self.low_dims))
                else:
                    encoder = self.get_proj_model(n_epoch-self.period)
                    prev_embedding = encoder(prev_data).cpu().numpy()
                alpha = find_alpha(prev_data, train_data, n_neighbors=15)
                alpha[alpha < 0.3] = 0.0 # alpha >=0.5 is convincing
                # update_dists = find_update_dist(prev_data, train_data, sigmas, rhos)
                # update_dists[update_dists < 0.05] = 0.0
                # alpha = alpha*update_dists
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
            # create embedding
            history = parametric_model.fit(
                edge_dataset,
                epochs=200,
                steps_per_epoch=steps_per_epoch,
                callbacks=callbacks,
                max_queue_size=100,
            )
            # save for later use
            parametric_model.prev_trainable_variables = weights_dict["prev"]
            flag = ""
            if self.advance_border_gen:
                flag = "_advance"

            if self.temporal:
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
            print("Average time for training visualzation model: {:.4f}".format(
                (t1 - t0) / int((self.epoch_end - self.epoch_start) / self.period + 1)))
        return (t1 - t0) / int((self.epoch_end - self.epoch_start) / self.period + 1)

    def get_proj_model(self, epoch_id):
        '''
        get the encoder of epoch_id
        :param epoch_id: int
        :return: encoder of epoch epoch_id
        '''
        flag = ""
        if self.advance_border_gen:
            flag = "_advance"
        if self.temporal:
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
        if self.advance_border_gen:
            flag = "_advance"
        if self.temporal:
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
        '''
        ba
        '''
        decoder = self.get_inv_model(epoch_id)
        if decoder is None:
            return None
        else:
            data = np.expand_dims(data, axis=0)
            representation_data = decoder(data).cpu().numpy()
            return representation_data.squeeze()

    def batch_inverse(self, data, epoch_id):
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
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        fc_model = torch.nn.Sequential(*(list(self.model.children())[self.split:]))
        data = data.to(self.device)
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
        self.model.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
        self.model = self.model.to(self.device)
        self.model.eval()

        repr_model = torch.nn.Sequential(*(list(self.model.children())[:self.split]))

        data = data.to(self.device)
        representation_data = batch_run(repr_model, data, self.repr_num)
        return representation_data

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

        data = self.testing_data
        data = self.get_representation_data(epoch_id, data)
        data = torch.from_numpy(data)
        data = data.to(self.device)
        pred = batch_run(fc_model, data, self.class_num)
        np.save(location, pred)
        return pred

    def get_epoch_train_repr_data(self, epoch_id):
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "train_data.npy")
        if os.path.exists(location):
            train_data = np.load(location)
            return train_data
        else:
            print("No data!")
            return None

    def get_epoch_train_labels(self, epoch_id):
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        if os.path.exists(index_file):
            index = load_labelled_data_index(index_file)
            labels = self.training_labels[index].cpu().numpy()
            return labels
        else:
            print("No data!")
            return None

    def get_epoch_test_repr_data(self, epoch_id):
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "test_data.npy")
        if os.path.exists(location):
            test_data = np.load(location)
            return test_data
        else:
            print("No data!")
            return None

    def get_epoch_test_labels(self, epoch_id):
        labels = self.testing_labels.cpu().numpy()
        return labels

    def get_epoch_plot_measures(self, epoch_id):
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
        border[diff < 0.1] = 1
        diff[border == 0] = 0

        mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(mesh_classes)
        color = self.cmap(mesh_classes / mesh_max_class)

        diff = diff.reshape(-1, 1)

        color = color[:, 0:3]
        color = diff * 0.5 * color + (1 - diff) * np.ones(color.shape, dtype=np.uint8)
        decision_view = color.reshape(resolution, resolution, 3)
        grid_view = grid.reshape(resolution, resolution, 2)
        return grid_view, decision_view

    def get_standard_classes_color(self):
        '''
        get background view
        :param epoch_id: epoch that need to be visualized
        :param resolution: background resolution
        :return:
            grid_view : numpy.ndarray, self.resolution,self.resolution, 2
            color : numpy.ndarray, self.resolution,self.resolution, 3
        '''
        mesh_max_class = self.class_num - 1
        mesh_classes = np.arange(10)
        color = self.cmap(mesh_classes / mesh_max_class)

        color = color[:, 0:3]
        return color

    def _s(self, is_for_frontend=False):
        '''
        Initialises matplotlib artists and plots. from DeepView
        '''
        # if self.interactive:
        #     plt.ion()
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

    def _visualize_init(self):
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

        # # labels = prediction
        # for c in range(self.class_num):
        #     color = self.cmap(c/(self.class_num-1))
        #     plot = self.ax.plot([], [], '.', label=self.classes[c], ms=1,
        #         color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
        #     self.sample_plots.append(plot[0])
        #
        # # labels != prediction, labels be a large circle
        # for c in range(self.class_num):
        #     color = self.cmap(c/(self.class_num-1))
        #     plot = self.ax.plot([], [], 'o', markeredgecolor=color,
        #         fillstyle='full', ms=3, mew=2.5, zorder=3)
        #     self.sample_plots.append(plot[0])
        #
        # # labels != prediction, prediction stays inside of circle
        # for c in range(self.class_num):
        #     color = self.cmap(c / (self.class_num - 1))
        #     plot = self.ax.plot([], [], '.', markeredgecolor=color,
        #                         fillstyle='full', ms=2, zorder=4)
        #     self.sample_plots.append(plot[0])
        #
        # # highlight
        # color = (0.0, 0.0, 0.0, 1.0)
        # plot = self.ax.plot([], [], 'o', markeredgecolor=color,
        #                     fillstyle='full', ms=1, mew=1, zorder=1)
        # self.sample_plots.append(plot[0])

        # train_data labels
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=1,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        # testing wrong labels
        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '^', markeredgecolor=color,
                fillstyle='full', ms=9, mew=9, zorder=1)
            self.sample_plots.append(plot[0])

        # highlight
        color = (0.0, 0.0, 0.0, 1.0)
        plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                            fillstyle='full', ms=2.0, mew=2.0, zorder=3)
        self.sample_plots.append(plot[0])

        # set the mouse-event listeners
        # self.fig.canvas.mpl_connect('pick_event', self.show_sample)
        # self.fig.canvas.mpl_connect('button_press_event', self.show_sample)
        self.disable_synth = False

    def show(self, epoch_id):
        '''
        Shows the current plot. from DeepView
        '''
        # if not hasattr(self, 'fig'):
        #     self._s()
        self._s()

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
            self.sample_plots[self.class_num + c].set_data(data.transpose())
        #
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
        self._s()

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

    def visualize(self, epoch_id, train_data, train_labels, test_data, test_labels, path, highlight_index):
        '''
        Shows the current plot.
        '''
        # if not hasattr(self, 'fig'):
        #     self._s()
        self._visualize_init()

        x_min, y_min, x_max, y_max = self.get_epoch_plot_measures(epoch_id)

        grid, decision_view = self.get_epoch_decision_view(epoch_id, self.resolution)
        self.cls_plot.set_data(decision_view)
        self.cls_plot.set_extent((x_min, x_max, y_max, y_min))
        self.ax.set_xlim((x_min, x_max))
        self.ax.set_ylim((y_min, y_max))

        # params_str = 'res: %d'
        # desc = params_str % (self.resolution)
        # self.desc.set_text(desc)

        # pred = self.get_pred(epoch_id, train_data)
        # pred = pred.argmax(axis=1)

        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(train_data).cpu().numpy()
        test_embedding = proj_encoder(test_data).cpu().numpy()
        # for c in range(self.class_num):
        #     data = embedding[np.logical_and(train_labels == c, train_labels == pred)]
        #     self.sample_plots[c].set_data(data.transpose())
        #
        # for c in range(self.class_num):
        #     data = embedding[np.logical_and(train_labels == c, train_labels != pred)]
        #     self.sample_plots[self.class_num + c].set_data(data.transpose())
        # #
        # for c in range(self.class_num):
        #     data = embedding[np.logical_and(pred == c, train_labels != pred)]
        #     self.sample_plots[2 * self.class_num + c].set_data(data.transpose())
        #
        # data = embedding[highlight_index]
        # self.sample_plots[3 * self.class_num].set_data(data.transpose())
        for c in range(self.class_num):
            data = embedding[train_labels == c]
            self.sample_plots[c].set_data(data.transpose())
        for c in range(self.class_num):
            data = test_embedding[test_labels == c]
            self.sample_plots[self.class_num + c].set_data(data.transpose())
        data = embedding[highlight_index]
        self.sample_plots[2*self.class_num].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # plt.text(-8, 8, "test", fontsize=18, style='oblique', ha='center', va='top', wrap=True)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)

    def batch_get_embedding(self, data, epoch_id):
        '''
        get embedding of subject model at epoch_id
        :param data: torch.Tensor
        :param epoch_id:
        :return: embedding, numpy.array
        '''
        repr_data = self.get_representation_data(epoch_id, data)
        embedding = self.batch_project(repr_data, epoch_id)
        return embedding

    # TODO test those functions

    def get_epoch_border_centers(self, epoch_id):
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

    def proj_nn_perseverance_knn_train(self, epoch_id, n_neighbors=15):
        train_data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(train_data).cpu().numpy()

        del encoder
        gc.collect()

        val = evaluate_proj_nn_perseverance_knn(train_data, embedding, n_neighbors, metric="euclidean")
        return val

    def proj_nn_perseverance_knn_test(self, epoch_id, n_neighbors=15):
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
        encoder = self.get_proj_model(epoch_id)
        border_centers = self.get_epoch_border_centers(epoch_id)
        train_data = self.get_epoch_train_repr_data(epoch_id)
        # test_data = self.get_representation_data(epoch_id, self.testing_data)
        test_data = self.get_epoch_test_repr_data(epoch_id)
        fitting_data = np.concatenate((train_data, test_data), axis=0)

        low_center = encoder(border_centers).cpu().numpy()
        low_data = encoder(fitting_data).cpu().numpy()

        del encoder
        gc.collect()

        val = evaluate_proj_boundary_perseverance_knn(fitting_data, low_data, border_centers, low_center, n_neighbors)

        return val

    def proj_temporal_perseverance_train(self, n_neighbors=15):
        l = len(self.training_labels)
        eval_num = int((self.epoch_end - self.epoch_start) / self.period)
        alpha = np.zeros((eval_num, l))
        delta_x = np.zeros((eval_num, l))
        for n_epoch in range(self.epoch_start+self.period, self.epoch_end+1, self.period):

            prev_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch - self.period), "train_data.npy")
            prev_data = np.load(prev_data_loc)
            encoder = self.get_proj_model(n_epoch - self.period)
            prev_embedding = encoder(prev_data).cpu().numpy()

            del encoder
            gc.collect()

            encoder = self.get_proj_model(n_epoch)
            data = self.get_epoch_train_repr_data(n_epoch)
            embedding = encoder(data).cpu().numpy()

            del encoder
            gc.collect()

            alpha_ = backend.find_alpha(prev_data, data, n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - embedding, axis=1)

            alpha[int((n_epoch - self.epoch_start) / self.period - 1)] = alpha_
            delta_x[int((n_epoch - self.epoch_start) / self.period - 1)] = delta_x_

        # val_entropy = evaluate_proj_temporal_perseverance_entropy(alpha, delta_x)
        val_corr = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)
        return val_corr

    def proj_temporal_perseverance_test(self, n_neighbors=15):
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

            alpha_ = backend.find_alpha(prev_data, data, n_neighbors)
            delta_x_ = np.linalg.norm(prev_embedding - embedding, axis=1)
            alpha[int((n_epoch - self.epoch_start) / self.period - 1)] = alpha_
            delta_x[int((n_epoch - self.epoch_start) / self.period - 1)] = delta_x_

        # val_entropy = evaluate_proj_temporal_perseverance_entropy(alpha, delta_x)
        val_corr = evaluate_proj_temporal_perseverance_corr(alpha, delta_x)

        return val_corr

    def inv_accu_train(self, epoch_id):
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

    def inv_logits_diff_train(self, epoch_id):
        data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        labels = self.get_epoch_train_labels(epoch_id)
        ori_pred = self.get_pred(epoch_id, data)
        new_pred = self.get_pred(epoch_id, inv_data)

        val = evaluate_inv_conf(labels, ori_pred, new_pred)
        return val

    def nn_pred_accu(self, epoch_id, n_neighbors=15):
        data = self.get_epoch_train_repr_data(epoch_id)
        n_trees = min(64, 5 + int(round((data.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(data.shape[0]))))
        nnd = NNDescent(
            data,
            n_neighbors=n_neighbors,
            metric="euclidean",
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            verbose=True
        )
        indices, _ = nnd.neighbor_graph
        pred = self.get_pred(epoch_id, data)
        pred = pred.argmax(axis=1)
        pers = pred[indices]
        res = np.zeros(len(pred))
        for i in range(len(pred)):
            res[i] = np.sum(pers[i] == pred[i])/n_neighbors
        return res.mean()

    def training_accu(self,epoch_id):
        train_data = self.get_epoch_train_repr_data(epoch_id)
        labels = self.get_epoch_train_labels(epoch_id)
        pred = self.get_pred(epoch_id, train_data).argmax(-1)
        val = evaluate_inv_accu(labels, pred)
        return val

    def testing_accu(self, epoch_id):
        # test_data = self.testing_data
        labels = self.testing_labels.cpu().numpy()
        # repr_data = self.get_representation_data(epoch_id, test_data)
        repr_data = self.get_epoch_test_repr_data(epoch_id)
        pred = self.get_pred(epoch_id, repr_data).argmax(-1)
        val = evaluate_inv_accu(labels, pred)
        return val

    def inv_nn_preserve_train(self, epoch_id, n_neighbors=15):
        data = self.get_epoch_train_repr_data(epoch_id)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        inv_data = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        val = evaluate_inv_nn(data, inv_data, n_neighbors)
        return val

    def inv_nn_preserve_test(self, epoch_id, n_neighbors=15):
        # test_data = self.get_representation_data(epoch_id, self.testing_data)
        test_data = self.get_epoch_test_repr_data(epoch_id)
        train_data = self.get_epoch_train_repr_data(epoch_id)

        fitting_data = np.concatenate((train_data, test_data), axis=0)
        encoder = self.get_proj_model(epoch_id)
        embedding = encoder(fitting_data).cpu().numpy()
        del encoder
        gc.collect()

        decoder = self.get_inv_model(epoch_id)
        recon = decoder(embedding).cpu().numpy()
        del decoder
        gc.collect()

        val = evaluate_inv_nn(fitting_data, recon, n_neighbors)
        return val

    def get_new_index(self, epoch_id):
        prev_epoch = epoch_id - self.period
        if prev_epoch < self.epoch_start:
            return list()

        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        index = load_labelled_data_index(index_file)
        prev_index_file = os.path.join(self.model_path, "Epoch_{:d}".format(prev_epoch), "index.json")
        prev_index = load_labelled_data_index(prev_index_file)
        new_I = len(index) - len(prev_index)

        return index[new_I:]

    def noisy_data_index(self):
        index_file = os.path.join(self.model_path, "index.json")
        if not os.path.exists(index_file):
            return list()
        return load_labelled_data_index(index_file)

    def get_epoch_index(self, epoch_id):
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        index = load_labelled_data_index(index_file)
        return index
