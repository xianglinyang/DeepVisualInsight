from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from Model.model import *

def prepare_data(content_path, data, start_demo_iter, end_demo_iter, resolution, direct_call=True):
    sys.path.append(content_path)
    prefix = ''
    if direct_call:
        prefix = 'server/'
    # net = resnet18()
    net = ResNet18()
    demo_iter_number = end_demo_iter - start_demo_iter

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    mms = MMS(content_path, net, 1, 200, 512, 10, classes, cmap="tab10", resolution=resolution, boundary_diff=1, neurons=128,
              verbose=1)
    # mms.data_preprocessing()
    # mms.prepare_visualization_for_all()

    dimension_reduction_result = np.zeros((demo_iter_number, data.shape[0], 2))
    '''
    for i in range(start_demo_iter, end_demo_iter):
        print("Epoch number:"+str(i))
        result = mms.batch_get_embedding(data, i)
        dimension_reduction_result[i - start_demo_iter] = result

    with open(prefix+'data/2D.npy', 'wb') as f:
        np.save(f, dimension_reduction_result)
    '''
    grid_list = np.zeros((demo_iter_number, resolution, resolution, 2))
    decision_view_list = np.zeros((demo_iter_number, resolution, resolution, 3))
    for i in range(start_demo_iter, end_demo_iter):
        grid, decision_view = mms.get_epoch_decision_view(i, resolution)
        grid_list[i - start_demo_iter] = grid
        decision_view_list[i - start_demo_iter] = decision_view

    with open(prefix+'data/grid.npy', 'wb') as f:
        np.save(f, grid_list)

    with open(prefix+'data/decision_view.npy', 'wb') as f:
        np.save(f, decision_view_list)

    color = mms.get_standard_classes_color()

    with open(prefix+'data/color.npy', 'wb') as f:
        np.save(f, color)

if __name__ == "__main__":
    content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
    testing_data = torch.load(
        "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10\\Testing_data\\testing_dataset_data.pth")
    prepare_data(content_path, testing_data, start_demo_iter=180, end_demo_iter=200, resolution=400)