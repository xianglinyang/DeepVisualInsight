from deepvisualinsight.MMS import MMS
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def prepare_data(content_path, data, iteration, resolution, folder_name, direct_call=True):
    sys.path.append(content_path)
    from Model.model import ResNet18
    prefix = folder_name + '/'
    if direct_call:
        prefix = 'server/'+folder_name + '/'
    # net = resnet18()
    net = ResNet18()
  

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    mms = MMS(content_path, net, 1, 200, 512, 10, classes, cmap="tab10", resolution=resolution, boundary_diff=1, neurons=128,
              verbose=1)
    # mms.data_preprocessing()
    # mms.prepare_visualization_for_all()

    dimension_reduction_result = mms.batch_get_embedding(data, iteration)

    with open(prefix+'dimension_reduction_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, dimension_reduction_result)

    grid, decision_view = mms.get_epoch_decision_view(iteration, resolution)


    with open(prefix+'grid_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, grid)

    with open(prefix+'decision_view_'+str(iteration)+'.npy', 'wb') as f:
        np.save(f, decision_view)

    color = mms.get_standard_classes_color()

    with open(prefix+'color.npy', 'wb') as f:
        np.save(f, color)

if __name__ == "__main__":
    content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
    testing_data = torch.load(
        "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10\\Testing_data\\testing_dataset_data.pth")
    prepare_data(content_path, testing_data, iteration=180, folder_name="data/entropy_tl_cifar10", resolution=400)
