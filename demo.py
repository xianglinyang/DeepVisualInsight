from deepvisualinsight.MMS import MMS
from deepvisualinsight import utils
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import json
import tensorflow as tf

content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\resnet18_fashionmnist"
# content_path = "E:\\DVI_exp_data\\resnet18_mnist"
# content_path = "E:\\DVI_exp_data\\active_learning\\random_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\inexpressive_model"
# content_path = "../../DVI_EXP/normal_training/resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\active_learning\\same_start\\random"
content_path = "E:\\DVI_exp_data\\active_learning\\same_start\\entropy"

sys.path.append(content_path)

from Model.model import *
# net = resnet18()
net = ResNet18()
# net = CIFAR_17()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
# classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


# TODO temporal loss dynamically change weight?
mms = MMS(content_path, net, 0, 0, 1, 512, 10, classes, cmap="tab10", resolution=100, neurons=256, verbose=1, temporal=False, split=-1, advance_border_gen=True, attack_device="cuda:0")
encoder_location = os.path.join("E:\\DVI_exp_data\\active_learning\\same_start\\random", "Model", "Epoch_{:d}".format(1), "encoder_advance")
encoder = tf.keras.models.load_model(encoder_location)
decoder_location = os.path.join("E:\\DVI_exp_data\\active_learning\\same_start\\random", "Model", "Epoch_{:d}".format(1), "decoder_advance")
decoder = tf.keras.models.load_model(decoder_location)

# mms.data_preprocessing()
# mms.prepare_visualization_for_all(encoder, decoder)
#
# for i in [40, 120, 200]:
#     print(mms.proj_nn_perseverance_knn_train(i, 10))
#     print(mms.proj_nn_perseverance_knn_train(i, 15))
#     print(mms.proj_nn_perseverance_knn_train(i, 30))
#
#     print(mms.proj_nn_perseverance_knn_test(i, 10))
#     print(mms.proj_nn_perseverance_knn_test(i, 15))
#     print(mms.proj_nn_perseverance_knn_test(i, 30))
#
#     print(mms.proj_boundary_perseverance_knn_train(i, 10))
#     print(mms.proj_boundary_perseverance_knn_train(i, 15))
#     print(mms.proj_boundary_perseverance_knn_train(i, 30))
#
#     print(mms.proj_boundary_perseverance_knn_test(i, 10))
#     print(mms.proj_boundary_perseverance_knn_test(i, 15))
#     print(mms.proj_boundary_perseverance_knn_test(i, 30))
#
#     print(mms.inv_accu_train(i))
#     print(mms.inv_accu_test(i))
#     print(mms.inv_conf_diff_train(i))
#     print(mms.inv_conf_diff_test(i))


# mms.proj_temporal_perseverance_train(15)
# mms.nn_pred_accu(200, 15)
# mms.inv_conf_diff_train(2)
# mms.inv_conf_diff_train(2)
# mms.inv_conf_diff_train(5)
# mms.inv_conf_diff_train(1)
# mms.nn_pred_accu(1, 15)
# print(1)
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

#
img_save_location = os.path.join(mms.content_path, "img")
if not os.path.exists(img_save_location):
    os.mkdir(img_save_location)

for i in range(0, 1, 1):
    train_data = mms.get_epoch_train_repr_data(i)
    labels = mms.get_epoch_train_labels(i)
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
    plt.savefig(os.path.join(img_save_location, "{:d}".format(i)))
    mms.savefig(i, os.path.join(img_save_location, "b_{:d}".format(i)))




# ## noisy model highlight points
# img_save_location = os.path.join(mms.content_path, "img")
# index_file = os.path.join(mms.content_path, "index.json")
# index = utils.load_labelled_data_index(index_file)
# ol_file = os.path.join(mms.content_path, "old_labels.json")
# old_labels = utils.load_labelled_data_index(ol_file)
# nl_file = os.path.join(mms.content_path, "new_labels.json")
# new_labels = utils.load_labelled_data_index(nl_file)
# data = torch.load("E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10\\Training_data\\training_dataset_data_.pth")
# labels = torch.load("E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10\\Training_data\\training_dataset_label_.pth").cpu().numpy()
#
# fd = data
# train_data = mms.get_representation_data(200, fd)
# pred = mms.get_pred(200, train_data).argmax(-1)
# l = labels
#
#
# for i in range(200, 201, 1):
#     train_data = mms.get_representation_data(i, data)
#     # labels = mms.get_pred(i, train_data).argmax(-1)
#
#     mms.visualize(i, train_data, labels, os.path.join(img_save_location, "hl_{:d}".format(i)), index)

# active learning
# img_save_location = os.path.join(mms.content_path, "img")
#
# for i in range(100, 200, 20):
#     # index_file = os.path.join(mms.model_path, "Epoch_{:d}".format(i), "index.json")
#     # index = utils.load_labelled_data_index(index_file)
#
#     new_index_file = os.path.join(mms.model_path, "Epoch_{:d}".format(i+1), "index.json")
#     new_index = utils.load_labelled_data_index(new_index_file)
#     training_data = mms.training_data[new_index]
#     training_labels = mms.training_labels[new_index].cpu().numpy()
#
#     testing_data = mms.testing_data
#     testing_labels = mms.testing_labels.cpu().numpy()
#
#     train_data = mms.get_representation_data(i, training_data)
#     test_data = mms.get_representation_data(i, testing_data)
#
#     pred = mms.get_epoch_test_pred(i).argmax(-1)
#     index = (pred != testing_labels)
#     test_data = test_data[index]
#     testing_labels = testing_labels[index]
#     mms.visualize(i, train_data, training_labels, test_data, testing_labels, os.path.join(img_save_location, "alb_{:d}".format(i)), np.arange(-1000, 0, 1))

