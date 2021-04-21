import unittest
from deepvisualinsight.MMS import MMS
import sys
import numpy as np
import torch

# diagram of code structure
# https://docs.google.com/document/d/1xGgBpFzCcwiRUU_l9_Y_q0nTeIcCG1-GvXWnBZhfcGs/edit
# https://github.com/xianglinyang/DeepVisualInsight/wiki/usage_instructions


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)

        # choose resnet18_cifar10 to test all apis
        content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
        sys.path.append(content_path)

        from Model.model import resnet18
        net = resnet18()
        # net = ResNet18()
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
        self.mms = MMS(content_path, net, 0, 200, 1, 512, 10, classes, cmap="tab10", resolution=100,
                                neurons=256, verbose=1, temporal=False, split=-1, advance_border_gen=True,
                                attack_device="cuda:0")

    ############################################### Backend API ###################################################
    '''get autoencoder model'''
    def test_get_proj_model(self):
        # if encoder does not exist
        encoder = self.mms.get_proj_model(-1)
        self.assertIsNone(encoder)

        # general setting
        encoder = self.mms.get_proj_model(120)
        self.assertIsNotNone(encoder)

    def test_get_env_model(self):
        # if decoder does not exist
        decoder = self.mms.get_inv_model(-1)
        self.assertIsNone(decoder)

        # general setting
        decoder = self.mms.get_inv_model(120)
        self.assertIsNotNone(decoder)

    '''Dimension reduction'''
    def test_batch_project(self):
        # general setting
        epoch_id = 120
        data = np.random.rand(500, 512)
        embedding = self.mms.batch_project(data, epoch_id)

        shape = (500, 2)
        output_shape = embedding.shape
        self.assertEqual(len(shape), len(output_shape))
        self.assertTupleEqual(shape, output_shape)

        # if projection function does not exist
        epoch_id = -1
        data = np.random.rand(500, 512)
        embedding = self.mms.batch_project(data, epoch_id)
        self.assertIsNone(embedding)

    def test_individual_project(self):
        # general setting
        epoch_id = 120
        data = np.random.rand(512)
        embedding = self.mms.individual_project(data, epoch_id)

        shape = (2,)
        output_shape = embedding.shape
        self.assertEqual(len(shape), len(output_shape))
        self.assertTupleEqual(shape, output_shape)

        # encoder does not exist
        epoch_id = -1
        data = np.random.rand(512)
        embedding = self.mms.individual_project(data, epoch_id)
        self.assertIsNone(embedding)

    def test_batch_inverse(self):
        # general setting
        epoch_id = 120
        data = np.random.rand(500, 2)
        recon = self.mms.batch_inverse(data, epoch_id)

        shape = (500, 512)
        output_shape = recon.shape
        self.assertEqual(len(shape), len(output_shape))
        self.assertTupleEqual(shape, output_shape)

        # if projection function does not exist
        epoch_id = -1
        data = np.random.rand(500, 2)
        recon = self.mms.batch_inverse(data, epoch_id)
        self.assertIsNone(recon)

    def test_individual_inverse(self):
        # general setting
        epoch_id = 120
        data = np.random.rand(2)
        recon = self.mms.individual_inverse(data, epoch_id)

        shape = (512,)
        output_shape = recon.shape
        self.assertEqual(len(shape), len(output_shape))
        self.assertTupleEqual(shape, output_shape)

        # encoder does not exist
        epoch_id = -1
        data = np.random.rand(2)
        recon = self.mms.individual_inverse(data, epoch_id)
        self.assertIsNone(recon)

    '''tool'''
    def test_get_incorrect_predictions(self):
        # model does not exist
        epoch_id = -1
        data = torch.rand((50, 512), dtype=torch.float)
        target = np.random.choice(10, 50)

        self.assertRaises(FileNotFoundError, self.mms.get_incorrect_predictions, epoch_id, data, target)
        with self.assertRaises(FileNotFoundError) as context:
            pred = self.mms.get_incorrect_predictions(epoch_id, data, target)
            self.assertIsNone(pred)
        # self.assertTrue("subject model does not exist" in context.exception)

        # general setting
        epoch_id = 120
        pred = self.mms.get_incorrect_predictions(epoch_id, data, target)
        shape = (50, )
        self.assertTupleEqual(shape, pred.shape)

    def test_get_representation_data(self):
        # model does not exist
        epoch_id = -1
        data = torch.rand((50, 3, 32, 32), dtype=torch.float)

        # self.assertRaises(FileNotFoundError, self.mms.get_incorrect_predictions, epoch_id, data, target)
        with self.assertRaises(FileNotFoundError) as context:
            repr_data = self.mms.get_representation_data(epoch_id, data)
            self.assertIsNone(repr_data)
        # self.assertTrue("subject model does not exist" in context.exception)

        # general setting
        epoch_id = 120
        pred = self.mms.get_representation_data(epoch_id, data)
        shape = (50, 512)
        self.assertTupleEqual(shape, pred.shape)

    def test_get_epoch_train_repr_data(self):
        epoch_id = 120
        repr_data = self.mms.get_epoch_train_repr_data(epoch_id)
        shape = (50000, 512)
        self.assertTupleEqual(shape, repr_data.shape)

    def test_get_epoch_test_repr_data(self):
        epoch_id = 120
        repr_data = self.mms.get_epoch_test_repr_data(epoch_id)
        shape = (10000, 512)
        self.assertTupleEqual(shape, repr_data.shape)

    def test_get_epoch_train_labels(self):
        epoch_id = 120
        labels = self.mms.get_epoch_train_labels(epoch_id)
        shape = (50000, )
        self.assertTupleEqual(shape, labels.shape)

    def test_get_epoch_test_labels(self):
        epoch_id = 120
        labels = self.mms.get_epoch_test_labels()
        shape = (10000, )
        self.assertTupleEqual(shape, labels.shape)

    def test_get_pred(self):
        epoch_id = 120
        data = np.random.rand(50, 512)
        shape = (50, 10)
        pred = self.mms.get_pred(epoch_id, data)
        self.assertTupleEqual(shape, pred.shape)

    def test_epoch_train_pred(self):
        epoch_id = 120
        pred = self.mms.get_epoch_train_pred(epoch_id)
        shape = (50000, 10)
        self.assertTupleEqual(pred.shape, shape)

    def test_epoch_test_pred(self):
        epoch_id = 120
        pred = self.mms.get_epoch_test_pred(epoch_id)
        shape = (10000, 10)
        self.assertTupleEqual(pred.shape, shape)

    def test_batch_embedding(self):
        data = torch.rand(50, 3, 32, 32)
        epoch_id = 120
        embedding = self.mms.batch_embedding(data, epoch_id)
        shape = (50, 2)
        self.assertTupleEqual(shape, embedding.shape)

    def test_get_epoch_border_centers(self):
        epoch_id = 120
        border_repr = self.mms.get_epoch_border_centers(epoch_id)
        shape = (5000, 512)
        self.assertTupleEqual(shape, border_repr.shape)

    ########################################## Evaluation Functions ###############################################
    '''project'''
    # # nn preserving property
    # val = mms.proj_nn_perseverance_knn_train(epoch_id, n_neighbors=15)
    # val = mms.proj_nn_perseverance_knn_test(epoch_id, n_neighbors=15)
    #
    # # boundary preserving property
    # val = mms.proj_boundary_perseverance_knn_train(epoch_id, n_neighbors=15)
    # val = mms.proj_boundary_perseverance_knn_test(epoch_id, n_neighbors=15)
    #
    # # temporal preserving property
    # val = mms.proj_boundary_perseverance_knn_train(epoch_id, n_neighbors=15)
    # val = mms.proj_boundary_perseverance_knn_test(epoch_id, n_neighbors=15)
    #
    '''inverse'''
    # # prediction accuracy
    # val = mms.inv_accu_train(epoch_id)
    # val = mms.inv_accu_test(epoch_id)
    #
    # # confidence difference
    # val = mms.inv_conf_diff_train(epoch_id)
    # val = mms.inv_conf_diff_test(epoch_id)
    #
    # # mse
    # val = mms.inv_dist_train(epoch_id)
    # val = mms.inv_dist_test(epoch_id)
    #
    # # single instance
    # pred, conf_diff = point_inv_preserve(epoch_id, data)
    #
    '''subject model'''
    # accu = mms.training_accu(epoch_id)
    # accu = mms.testing_accu(epoch_id)
    ############################################# Visualization ###################################################
    # pass
    ########################################## Case Studies Related ###############################################
    '''active learning'''
    # idxs = mms.get_new_index(epoch_id)
    # idxs = mms.get_epoch_index(epoch_id)
    #
    '''noise data(mislabelled)'''
    # idxs = mms.noisy_data_index()
    # labels = mms.get_original_labels()


if __name__ == '__main__':
    unittest.main()
