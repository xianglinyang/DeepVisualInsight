from deepvisualinsight.MMS import MMS
from deepvisualinsight import utils
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import json

content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\resnet18_fashionmnist"
# content_path = "E:\\DVI_exp_data\\resnet18_mnist"
# content_path = "E:\\DVI_exp_data\\active_learning\\random_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10"

# measurement_save_location = os.path.join(content_path, "exp")
# if not os.path.exists(measurement_save_location):
#     os.mkdir(measurement_save_location)

sys.path.append(content_path)

from Model.model import *
net = resnet18()
# net = ResNet18()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
# classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


# TODO temporal loss dynamically change weight?
mms = MMS(content_path, net, 1, 200, 512, 10, classes, cmap="tab10", resolution=100, neurons=256, verbose=1, temporal=False)
# # v1,v2 = mms.proj_temporal_perseverance_train(10)
# # print(v1,v2)
# for i in range(100,200,20):
#     print("training_acc {}:".format(i), mms.training_accu(i))
#     print("testing_acc {}:".format(i), mms.testing_accu(i))

# result = np.zeros((200, 22))
#
# for i in range(1, 201, 1):
#     print(i)
#     result[i-1][0] = mms.proj_nn_perseverance_knn_train(i, 10)
#     result[i-1][1] = mms.proj_nn_perseverance_knn_train(i, 20)
#     result[i-1][2] = mms.proj_nn_perseverance_knn_train(i, 30)
#     result[i-1][3] = mms.proj_nn_perseverance_knn_train(i, 50)
#     result[i-1][4] = mms.proj_nn_perseverance_knn_test(i, 10)
#     result[i-1][5] = mms.proj_nn_perseverance_knn_test(i, 20)
#     result[i-1][6] = mms.proj_nn_perseverance_knn_test(i, 30)
#     result[i-1][7] = mms.proj_nn_perseverance_knn_test(i, 50)
#
#     result[i-1][8] = mms.proj_boundary_perseverance_knn_train(i, 10)
#     result[i-1][9] = mms.proj_boundary_perseverance_knn_train(i, 20)
#     result[i-1][10] = mms.proj_boundary_perseverance_knn_train(i, 30)
#     result[i-1][11] = mms.proj_boundary_perseverance_knn_train(i, 50)
#     result[i-1][12] = mms.proj_boundary_perseverance_knn_test(i, 10)
#     result[i-1][13] = mms.proj_boundary_perseverance_knn_test(i, 20)
#     result[i-1][14] = mms.proj_boundary_perseverance_knn_test(i, 30)
#     result[i-1][15] = mms.proj_boundary_perseverance_knn_test(i, 50)
#
#     result[i-1][16] = mms.inv_accu_train(i)
#     result[i-1][17] = mms.inv_accu_test(i)
#     result[i-1][18] = mms.inv_conf_diff_train(i)
#     result[i-1][19] = mms.inv_conf_diff_test(i)
#     result[i-1][20] = mms.inv_dist_train(i)
#     result[i-1][21] = mms.inv_dist_test(i)
# np.save(os.path.join(measurement_save_location, "exp.npy"), result)
