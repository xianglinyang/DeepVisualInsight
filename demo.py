from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# content_path = "E:\\DVI_exp_data\\resnet18"
content_path = "E:\\DVI_exp_data\\active_learning\\random_tl"
sys.path.append(content_path)

from Model.model import *
# net = resnet18()
net = ResNet18()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

mms = MMS(content_path, net, 1, 200, 512, 10, classes, cmap="tab10", resolution=100, boundary_diff=1, neurons=128, verbose=1)
mms.data_preprocessing()
mms.prepare_visualization_for_all()

# grid, decision_view = mms.get_epoch_decision_view(180)
# mms.show(181)
# img_save_location = os.path.join(mms.content_path, "img")
# if not os.path.exists(img_save_location):
#     os.mkdir(img_save_location)
# for i in range(1, 184, 1):
#     train_data = mms.get_epoch_repr_data(i)
#     labels = mms.get_epoch_labels(i)
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
#     plt.savefig(os.path.join(img_save_location,"{:d}".format(i)))
