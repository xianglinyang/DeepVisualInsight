import tensorflow as tf
from deepvisualinsight.utils import *
from deepvisualinsight.backend import *
import matplotlib.pyplot as plt
import matplotlib as mpl
# from deepvisualinsight import evaluate

class MMS:
    def __init__(self, content_path, model_structure, epoch_start, epoch_end, repr_num, class_num, classes, low_dims=2,
                 cmap="tab10", resolution=100, boundary_diff=1.5, neurons=None, temporal=False, verbose=1):
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
        boundary_diff : float
            the difference between top1 and top2 logits from last layer. The difference is used to define boundary.
        neurons : int
            the number of units inside each layer of autoencoder
        temporal: boolean, by default False
            choose whether to add temporal loss or not
        verbose : int, by default 1
        '''
        self.model = model_structure
        self.visualization_models = None
        self.subject_models = None
        self.content_path = content_path
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.training_data = None
        self.data_epoch_index = None
        self.testing_data = None
        self.repr_num = repr_num
        self.low_dims = low_dims
        self.cmap = plt.get_cmap(cmap)
        self.resolution = resolution
        self.boundary_diff = boundary_diff
        self.class_num = class_num
        self.classes = classes
        self.temporal = temporal
        self.verbose = verbose
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tf_device = tf.config.list_physical_devices('GPU')[0]
        tf.config.experimental.set_memory_growth(self.tf_device, True)
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
        for n_epoch in range(self.epoch_start, self.epoch_end+1, 1):
            index_file = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "index.json")
            index = load_labelled_data_index(index_file)
            training_data = self.training_data[index]
            training_labels = self.training_labels[index]

            model_location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "subject_model.pth")
            self.model.load_state_dict(torch.load(model_location))
            self.model = self.model.to(self.device)

            repr_model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

            n_clusters = math.floor(len(index) / 10)

            border_points = get_border_points(training_data, training_labels, self.model, self.boundary_diff)
            border_points = torch.from_numpy(border_points)
            border_points = border_points.to(self.device)
            border_representation = batch_run(repr_model, border_points, self.repr_num)
            border_centers = clustering(border_representation, n_clusters, verbose=0)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "border_centers.npy")
            np.save(location, border_centers)

            train_data = training_data.to(self.device)
            train_data_representation = batch_run(repr_model, train_data, self.repr_num)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_data.npy")
            np.save(location, train_data_representation)

            train_centers = clustering(train_data_representation, n_clusters, verbose=0)
            location = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_centers.npy")
            np.save(location, train_centers)
            if self.verbose > 0:
                print("Finish data preprocessing for Epoch {:d}...".format(n_epoch))

    def prepare_visualization_for_all(self):
        """
        conduct transfer learning to save the visualization model for each epoch
        """
        dims = (self.repr_num,)
        n_components = 2
        batch_size = 200
        self.encoder, self.decoder = define_autoencoder(dims, n_components, self.neurons)
        optimizer = tf.keras.optimizers.Adam(1e-3)
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=10 ** -2,
                patience=10,
                verbose=1,
            ),
            tf.keras.callbacks.LearningRateScheduler(define_lr_schedule),
        ]
        parametric_model = define_model(dims, self.low_dims, self.encoder, self.decoder, self.temporal)

        # self.data_preprocessing()
        for n_epoch in range(self.epoch_start, self.epoch_end+1, 1):
            losses, loss_weights = define_losses(200, n_epoch, self.epoch_end-self.epoch_start+1, self.temporal)
            parametric_model.compile(
                optimizer=optimizer, loss=losses, loss_weights=loss_weights,
            )

            train_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_centers.npy")
            border_centers_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "border_centers.npy")
            train_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "train_data.npy")

            # in case no data save for vis
            if not os.path.exists(train_data_loc):
                continue

            train_centers = np.load(train_centers_loc)
            border_centers = np.load(border_centers_loc)
            train_data = np.load(train_data_loc)

            complex = fuzzy_complex(train_data, 15)
            bw_complex = boundary_wise_complex(train_centers, border_centers, 15)
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
                    50,
                    batch_size,
                    parametric_embedding=True,
                    parametric_reconstruction=True,
                )
            else:
                prev_data_loc = os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch-1), "train_data.npy")
                if os.path.exists(prev_data_loc):
                    prev_data = np.load(prev_data_loc)
                else:
                    prev_data = None
                if prev_data is None:
                    prev_embedding = np.zeros((len(prev_data), self.low_dims))
                else:
                    encoder = self.get_proj_model(n_epoch-1)
                    prev_embedding = encoder(prev_data).cpu().numpy()
                alpha = find_alpha(prev_data, train_data, n_neighbors=15)
                (
                    edge_dataset,
                    batch_size,
                    n_edges,
                    edge_weight,
                ) = construct_temporal_mixed_edge_dataset(
                    (train_data, train_centers, border_centers),
                    complex,
                    bw_complex,
                    50,
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

            self.encoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "encoder"))
            self.decoder.save(os.path.join(self.model_path, "Epoch_{:d}".format(n_epoch), "decoder"))

            if self.verbose > 0:
                print("save visualized model for Epoch {:d}".format(n_epoch))

    def get_proj_model(self, epoch_id):
        '''
        get the encoder of epoch_id
        :param epoch_id: int
        :return: encoder of epoch epoch_id
        '''
        encoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "encoder")
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
        decoder_location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "decoder")
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
        self.model.load_state_dict(torch.load(model_location))
        self.model = self.model.to(self.device)

        fc_model = torch.nn.Sequential(*(list(self.model.children())[-1:]))
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
        self.model.load_state_dict(torch.load(model_location))
        self.model = self.model.to(self.device)

        repr_model = torch.nn.Sequential(*(list(self.model.children())[:-1]))

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
        self.model.load_state_dict(torch.load(model_location))
        self.model = self.model.to(self.device)

        fc_model = torch.nn.Sequential(*(list(self.model.children())[-1:]))

        data = torch.from_numpy(data)
        data = data.to(self.device)
        pred = batch_run(fc_model, data, self.class_num)
        return pred

    def get_epoch_repr_data(self, epoch_id):
        location = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "train_data.npy")
        if os.path.exists(location):
            train_data = np.load(location)
            return train_data
        else:
            print("No data!")
            return None

    def get_epoch_labels(self, epoch_id):
        index_file = os.path.join(self.model_path, "Epoch_{:d}".format(epoch_id), "index.json")
        if os.path.exists(index_file):
            index = load_labelled_data_index(index_file)
            labels = self.training_labels[index].cpu().numpy()
            return labels
        else:
            print("No data!")
            return None

    def get_epoch_plot_measures(self, epoch_id):
        train_data = self.get_epoch_repr_data(epoch_id)
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
        diff = sort_preds[:, -1] - sort_preds[:, -2]
        border = np.zeros(len(diff), dtype=np.uint8) + 0.05
        border[diff < self.boundary_diff] = 1
        diff[border == 0] = 0

        mesh_classes = mesh_preds.argmax(axis=1)
        mesh_max_class = max(mesh_classes)
        color = self.cmap(mesh_classes / mesh_max_class)

        diff = (diff / diff.max()).reshape(-1, 1)

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

        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], '.', label=self.classes[c], ms=1,
                color=color, zorder=2, picker=mpl.rcParams['lines.markersize'])
            self.sample_plots.append(plot[0])

        for c in range(self.class_num):
            color = self.cmap(c/(self.class_num-1))
            plot = self.ax.plot([], [], 'o', markeredgecolor=color,
                fillstyle='none', ms=3, mew=2.5, zorder=1)
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

        train_data = self.get_epoch_repr_data(epoch_id)
        train_labels = self.get_epoch_labels(epoch_id)
        pred = self.get_pred(epoch_id, train_data)
        pred = pred.argmax(axis=1)

        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(train_data).cpu().numpy()
        for c in range(self.class_num):
            data = embedding[train_labels == c]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(pred==c, train_labels!=c)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())

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

        train_data = self.get_epoch_repr_data(epoch_id)
        train_labels = self.get_epoch_labels(epoch_id)
        pred = self.get_pred(epoch_id, train_data)
        pred = pred.argmax(axis=1)

        proj_encoder = self.get_proj_model(epoch_id)
        embedding = proj_encoder(train_data).cpu().numpy()
        for c in range(self.class_num):
            data = embedding[train_labels == c]
            self.sample_plots[c].set_data(data.transpose())

        for c in range(self.class_num):
            data = embedding[np.logical_and(pred==c, train_labels!=c)]
            self.sample_plots[self.class_num+c].set_data(data.transpose())

        if os.name == 'posix':
            self.fig.canvas.manager.window.raise_()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.savefig(path)

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
    #
    # def proj_nn_perseverance_knn_train(self, epoch_id):
    #     train_data = self.get_epoch_repr_data(epoch_id)
    #     encoder = self.get_proj_model(epoch_id)
    #     embedding = encoder(train_data).cpu().numpy()
    #
    #     val = evaluate.evaluate_proj_nn_perseverance_knn(train_data, embedding, 15, metric="euclidean")
    #     return val

    # def proj_nn_perseverance_knn_test(self, epoch_id):