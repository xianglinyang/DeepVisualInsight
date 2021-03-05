from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\resnet18_fashionmnist"
# content_path = "E:\\DVI_exp_data\\resnet18_mnist"
# content_path = "E:\\DVI_exp_data\\active_learning\\random_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10"

sys.path.append(content_path)

from Model.model import *
net = resnet18()
# net = ResNet18()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
# classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


# TODO temporal loss dynamically change weight?
mms = MMS(content_path, net, 1, 200, 512, 10, classes, cmap="tab10", resolution=100, boundary_diff=1, neurons=256, verbose=1, temporal=True)
# mms.data_preprocessing()
# mms.prepare_visualization_for_all()

# print(mms.inv_accu_train(1))
# for i in range(0, 11, 1):
#     print(mms.nn_pred_accu(i, n_neighbors=15))
#     print(mms.inv_accu_train(i))
#     print(mms.inv_accu_test(i))

# print(mms.nn_pred_accu(50))
# print(mms.nn_pred_accu(80))
# print(mms.nn_pred_accu(100))
# print(mms.nn_pred_accu(200))
# print(mms.inv_accu_test(1))
# print(mms.inv_dist_train(1))
# print(mms.inv_dist_test(1))
# print(mms.inv_accu_train(20))
# print(mms.inv_accu_test(20))
# print(mms.inv_dist_train(20))
# print(mms.inv_dist_test(20))
# for i in range(1, 201, 1):
#     # print(mms.proj_nn_perseverance_knn_train(i, 15))
#     # print(mms.proj_nn_perseverance_knn_test(i, 15))
#     # print(mms.proj_boundary_perseverance_knn_train(i, 15))
#     # print(mms.proj_boundary_perseverance_knn_test(i, 15))
#     # print(mms.proj_temporal_perseverance_train(15))
#     # print(mms.proj_temporal_perseverance_test(15))
#     print(mms.inv_accu_train(i))
#     print(mms.inv_accu_test(i))
    # print(mms.inv_conf_diff_train(i))
    # print(mms.inv_conf_diff_test(i))
    # print(mms.inv_dist_train(i))
    # print(mms.inv_dist_test(i))


img_save_location = os.path.join(mms.content_path, "img")
if not os.path.exists(img_save_location):
    os.mkdir(img_save_location)
for i in range(1, 21, 1):
    train_data = mms.get_epoch_repr_data(i)
    labels = mms.get_epoch_labels(i)
    # with open("E:\\DVI_exp_data\\noisy_model\\resnet18\\index.json", 'r') as f:
    #     index = json.load(f)
    # with open("E:\\DVI_exp_data\\noisy_model\\resnet18\\new_labels.json", 'r') as f:
    #     ori_labels = json.load(f)
    if train_data is None:
        continue
    z = mms.batch_project(train_data, i)

    fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
    sc = ax.scatter(
        z[:, 0],
        z[:, 1],
        c=labels,
        cmap="tab10",
        s=0.1,
        alpha=0.5,
        rasterized=True,
    )
    ax.axis('equal')
    ax.set_title("parametric UMAP autoencoder embeddings-training data", fontsize=20)
    plt.savefig(os.path.join(img_save_location, "temporal_{:d}".format(i)))
    mms.savefig(i, os.path.join(img_save_location, "temporal_b_{:d}".format(i)))
