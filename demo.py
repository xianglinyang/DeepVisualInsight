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

# content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\resnet18_fashionmnist"
# content_path = "E:\\DVI_exp_data\\resnet18_mnist"
# content_path = "E:\\DVI_exp_data\\active_learning\\same_start\\random"
# content_path = "E:\\DVI_exp_data\\active_learning\\same_start\\entropy"
# content_path = "E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\inexpressive_model"
# content_path = "E:\\DVI_exp_data\\active_learning\\WAAL\\resnet_dis_5"
# content_path = "E:\\DVI_exp_data\\active_learning\\VAAL"
# content_path = "E:\\DVI_exp_data\\active_learning\\Coreset"
# content_path = "E:\\DVI_exp_data\\active_learning\\LeastConfident"
# content_path = "E:\\DVI_exp_data\\active_learning\\baseline\\random\\resnet18\\CIFAR10"
# content_path = "E:\\DVI_exp_data\\active_learning\\baseline\\LeastConfidence\\resnet18\\CIFAR10"
# content_path = "E:\\DVI_exp_data\\active_learning\\baseline\\coreset\\resnet18\\CIFAR10"
# content_path = "E:\\DVI_exp_data\\active_learning\\baseline\\LL4AL\\resnet18\\CIFAR10"
# content_path = "E:\\DVI_exp_data\\adv_training\\PGD.resnet18_cifar10"
# start_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\RQ1\\withoutB"
content_path = "E:\\DVI_exp_data\\RQ1\\test"

sys.path.append(content_path)

from Model.model import *
net = resnet18()
# net = ResNet18()
# net = CIFAR_17()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
# classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
# classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")


# TODO temporal loss dynamically change weight?
mms = MMS(content_path, net, 100, 100, 1, 512, 10, classes, cmap="tab10", resolution=100, neurons=256, verbose=1, temporal=False, split=-1, advance_border_gen=True, alpha=0.6, withoutB=False, attack_device="cuda:0")
# encoder_location = os.path.join(start_path, "Model", "Epoch_{:d}".format(100), "encoder_advance")
# encoder = tf.keras.models.load_model(encoder_location)
# decoder_location = os.path.join(start_path, "Model", "Epoch_{:d}".format(100), "decoder_advance")
# decoder = tf.keras.models.load_model(decoder_location)
# a = mms.get_uncertainty_score(3)
# b= mms.get_diversity_score(4)
# mms.data_preprocessing()
mms.prepare_visualization_for_all()

'''evaluate all properties'''
# for i in [40, 120, 200]:
#     t0 = time.time()
#     print(mms.proj_nn_perseverance_knn_train(i, 10))
#     print(mms.proj_nn_perseverance_knn_train(i, 15))
#     print(mms.proj_nn_perseverance_knn_train(i, 30))
#
#     t1 = time.time()
#     print((t1-t0)/3, "proj_nn_perseverance_knn_train")
#     print(mms.proj_nn_perseverance_knn_test(i, 10))
#     print(mms.proj_nn_perseverance_knn_test(i, 15))
#     print(mms.proj_nn_perseverance_knn_test(i, 30))
#
#     t2 = time.time()
#     print((t2 - t1) / 3, "proj_nn_perseverance_knn_test")
#     print(mms.proj_boundary_perseverance_knn_train(i, 10))
#     print(mms.proj_boundary_perseverance_knn_train(i, 15))
#     print(mms.proj_boundary_perseverance_knn_train(i, 30))
#
#     t3 = time.time()
#     print((t3 - t2) / 3, "boundary_perseverance_knn_train")
#     print(mms.proj_boundary_perseverance_knn_test(i, 10))
#     print(mms.proj_boundary_perseverance_knn_test(i, 15))
#     print(mms.proj_boundary_perseverance_knn_test(i, 30))
#
#     t4 = time.time()
#     print((t4 - t3) / 3, "boundary_perseverance_knn_test")
#     print(mms.inv_accu_train(i))
#     print(mms.inv_accu_test(i))
#
#     t5 = time.time()
#     print((t5 - t4) / 2, "inv_accu")
#     print(mms.inv_conf_diff_train(i))
#     print(mms.inv_conf_diff_test(i))
#     t6 = time.time()
#     print((t6 - t5) / 2, "inv_conf_diff")

# test temporal preserverance
# print(mms.proj_temporal_perseverance_train(15))
# print(mms.proj_temporal_perseverance_test(15))


'''visualization by plt'''
# img_save_location = os.path.join(mms.content_path, "img")
# if not os.path.exists(img_save_location):
#     os.mkdir(img_save_location)
#
# for i in range(10, 210, 10):
#     train_data = mms.get_epoch_train_repr_data(i)
#     labels = mms.get_epoch_train_labels(i)
#     # with open("E:\\DVI_exp_data\\noisy_model\\resnet18\\index.json", 'r') as f:
#     #     index = json.load(f)
#     # with open("E:\\DVI_exp_data\\noisy_model\\resnet18\\new_labels.json", 'r') as f:
#     #     ori_labels = json.load(f)
#     if train_data is None:
#         continue
#     z = mms.batch_project(train_data, i)
#
#     fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
#     sc = ax.scatter(
#         z[:, 0],
#         z[:, 1],
#         c=labels,
#         cmap="tab10",
#         s=0.1,
#         alpha=0.5,
#         rasterized=True,
#     )
#     ax.axis('equal')
#     ax.set_title("parametric UMAP autoencoder embeddings-training data", fontsize=20)
#     plt.savefig(os.path.join(img_save_location, "{:d}".format(i)))
#     mms.savefig(i, os.path.join(img_save_location, "b_{:d}".format(i)))


'''noisy model highlight points'''
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
#
# fd = data
# train_data = mms.get_representation_data(200, fd)
# pred = mms.get_pred(200, train_data).argmax(-1)
# l = labels
#
# for i in range(200, 201, 1):
#     train_data = mms.get_representation_data(i, data)
#     # labels = mms.get_pred(i, train_data).argmax(-1)
#     mms.customize_visualize(i, train_data, labels, None, None, os.path.join(img_save_location, "hl_{:d}".format(i)),
#                             index)

'''active learning(customized version)'''
# img_save_location = os.path.join(mms.content_path, "img")
#
# for i in range(20, 200, 20):
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
#
#     mms.customize_visualize(i, train_data, training_labels, test_data, testing_labels,
#                             os.path.join(img_save_location, "alb_{:d}".format(i)), np.arange(-1000, 0, 1))

'''active learning visualization'''
# img_save_location = os.path.join(mms.content_path, "img")
# if not os.path.exists(img_save_location):
#     os.mkdir(img_save_location)
#
# for i in range(1, 10, 1):
#     new_index_file = os.path.join(mms.model_path, "Epoch_{:d}".format(i+1), "index.json")
#     new_index = utils.load_labelled_data_index(new_index_file)
#     training_data = mms.training_data[new_index]
#     training_labels = mms.training_labels[new_index].cpu().numpy()
#     train_data = mms.get_representation_data(i, training_data)
#
#     index_file = os.path.join(mms.model_path, "Epoch_{:d}".format(i), "index.json")
#     index = utils.load_labelled_data_index(index_file)
#     l = []
#     for j in range(len(new_index)):
#         if new_index[j] not in index:
#             l.append(j)
#     mms.al_visualize(i, train_data, training_labels, os.path.join(img_save_location, "alb_{:d}".format(i)), l)

