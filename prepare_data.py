from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# content_path = "E:\\DVI_exp_data\\resnet18"

content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
sys.path.append(content_path)

from Model.model import *
# net = resnet18()
net = ResNet18()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

mms = MMS(content_path, net, 1, 200, 512, 10, classes, cmap="tab10", resolution=100, boundary_diff=1, neurons=128, verbose=1)
# mms.data_preprocessing()
# mms.prepare_visualization_for_all()

dimension_reduction_result = np.zeros((20, 10000, 2))
testing_data = torch.load( "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10\\Testing_data\\testing_dataset_data.pth")

for i in range(180, 200):
    result = mms.batch_get_embedding(testing_data, i)
    dimension_reduction_result[i - 180] = result

with open('server/data/2D.npy', 'wb') as f:
    np.save(f, dimension_reduction_result)

grid_list = np.zeros((20, 400, 400, 2))
decision_view_list = np.zeros((20, 400, 400, 3))
for i in range(180, 200):
    grid, decision_view = mms.get_epoch_decision_view(i, 400)
    grid_list[i-180] = grid
    decision_view_list[i-180] = decision_view

with open('server/data/grid.npy', 'wb') as f:
    np.save(f, grid_list)

with open('server/data/decision_view.npy', 'wb') as f:
    np.save(f, decision_view_list)

color = mms.get_standard_classes_color()

with open('server/data/color.npy', 'wb') as f:
    np.save(f, color)


#grid, decision_view = mms.get_epoch_standard_view(180, 100)

#print(grid.shape, decision_view.shape)
#grid, decision_view = mms.get_epoch_decision_view(180, 100)
#print(grid.shape, decision_view.shape)
#testing_data = torch.load("E:\\DVI_exp_data\\active_learning\\entropy_tl\\Testing_data\\testing_dataset_data.pth")



#grid, standard_view = mms.get_epoch_standard_view(180, 100)

# mms.savefig(181, "t.png")
# img_save_location = os.path.join(mms.content_path, "img")
# if not os.path.exists(img_save_location):
#     os.mkdir(img_save_location)
# for i in range(183, 201, 1):
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
#     mms.savefig(i, os.path.join(img_save_location,"b_{:d}".format(i)))
