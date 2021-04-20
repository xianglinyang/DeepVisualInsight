import unittest
from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import time


class MyTestCase(unittest.TestCase):
    def test_noisy_data_specific_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/noisy"
        training_data = torch.load("/models/data/noisy/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/noisy/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        noisy_data = mms.noisy_data_index()
        original_label = mms.get_original_labels()
        end = time.time()
        print("Succeed for noisy data related API, used time:"+str(end - start))

    def test_evaluation_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/noisy"
        training_data = torch.load("/models/data/noisy/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/noisy/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        evaluation = {}
        evaluation['nn_train_10'] = mms.proj_nn_perseverance_knn_train(iteration, 10)
        evaluation['nn_train_15'] = mms.proj_nn_perseverance_knn_train(iteration, 15)
        evaluation['nn_train_30'] = mms.proj_nn_perseverance_knn_train(iteration, 30)
        evaluation['nn_test_10'] = mms.proj_nn_perseverance_knn_test(iteration, 10)
        evaluation['nn_test_15'] = mms.proj_nn_perseverance_knn_test(iteration, 15)
        evaluation['nn_test_30'] = mms.proj_nn_perseverance_knn_test(iteration, 30)
        evaluation['bound_train_10'] = mms.proj_boundary_perseverance_knn_train(iteration, 10)
        evaluation['bound_train_15'] = mms.proj_boundary_perseverance_knn_train(iteration, 15)
        evaluation['bound_train_30'] = mms.proj_boundary_perseverance_knn_train(iteration, 30)
        evaluation['bound_test_10'] = mms.proj_boundary_perseverance_knn_test(iteration, 10)
        evaluation['bound_test_15'] = mms.proj_boundary_perseverance_knn_test(iteration, 15)
        evaluation['bound_test_30'] = mms.proj_boundary_perseverance_knn_test(iteration, 30)
        evaluation['inv_nn_train_10'] = mms.inv_nn_preserve_train(iteration, 10)
        evaluation['inv_nn_train_15'] = mms.inv_nn_preserve_train(iteration, 15)
        evaluation['inv_nn_train_30'] = mms.inv_nn_preserve_train(iteration, 30)
        evaluation['inv_nn_test_10'] = mms.inv_nn_preserve_test(iteration, 10)
        evaluation['inv_nn_test_15'] = mms.inv_nn_preserve_test(iteration, 15)
        evaluation['inv_nn_test_30'] = mms.inv_nn_preserve_test(iteration, 30)
        evaluation['inv_acc_train'] = mms.inv_accu_train(iteration)
        evaluation['inv_acc_test'] = mms.inv_accu_test(iteration)
        evaluation['inv_conf_train'] = mms.inv_conf_diff_train(iteration)
        evaluation['inv_conf_test'] = mms.inv_conf_diff_test(iteration)
        end = time.time()
        print("Succeed for evaluation related API, used time:"+str(end - start))

    def test_gap_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/noisy"
        training_data = torch.load("/models/data/noisy/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/noisy/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        gap_layer_data = mms.get_representation_data(iteration, data)
        prediction = mms.get_pred(iteration, gap_layer_data).argmax(-1)
        end = time.time()
        print("Succeed for gap layer related API, used time:"+str(end - start))

    def test_dimensional_reduction_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/noisy"
        training_data = torch.load("/models/data/noisy/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/noisy/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        dimension_reduction_result = mms.batch_embedding(data, iteration)
        end = time.time()
        print("Succeed for dimensional reduction related API, used time:"+str(end - start))

    def test_decision_space_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/noisy"
        training_data = torch.load("/models/data/noisy/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/noisy/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        grid, decision_view = mms.get_epoch_decision_view(iteration, resolution)
        end = time.time()
        print("Succeed for decision space related API, used time:"+str(end - start))

    def test_standard_color_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/noisy"
        training_data = torch.load("/models/data/noisy/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/noisy/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        color = mms.get_standard_classes_color()
        end = time.time()
        print("Succeed for standard color related API, used time:"+str(end - start))

    def test_active_learning_specific_timing(self):
        # parse request
        # forward request to mms
        # achieve runtime profile
        content_path = "/models/data/random"
        training_data = torch.load("/models/data/random/data/training_dataset_data.pth")
        testing_data = torch.load("/models/data/random/data/testing_dataset_data.pth")
        data = torch.cat((training_data, testing_data), 0)
        iteration = 1
        folder_name = "data/noisy"
        resolution = 200

        sys.path.append(content_path)
        net = None
        try:
            from Model.model import resnet18
            net = resnet18()
        except Exception as e:
            from Model.model import ResNet18
            net = ResNet18()

        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

        mms = MMS(content_path, net, 1, 11, 1, 512, 10, classes, cmap="tab10", resolution=resolution, neurons=256,
                      verbose=1)

        start = time.time()
        new_index = mms.get_new_index(iteration)
        current_index = mms.get_epoch_index(iteration)
        end = time.time()
        print("Succeed for active learning related API, used time:"+str(end - start))

if __name__ == '__main__':
    unittest.main()
