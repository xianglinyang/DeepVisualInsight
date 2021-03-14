from deepvisualinsight.MMS import MMS
from deepvisualinsight import utils
import sys
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import json
import argparse

content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\resnet18_fashionmnist"
# content_path = "E:\\DVI_exp_data\\resnet18_mnist"
# content_path = "E:\\DVI_exp_data\\active_learning\\random_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\active_learning\\entropy_tl_cifar10"
# content_path = "E:\\DVI_exp_data\\noisy_model\\resnet18_cifar10"
# content_path = "E:\\DVI_exp_data\\inexpressive_model"
# content_path = "../../DVI_EXP/normal_training/resnet18_cifar10"

def main(args):

    content_path = args.content_path
    sys.path.append(content_path)

    from Model.model import resnet18
    net = resnet18()
    # net = ResNet18()
    # net = CIFAR_17()

    classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    # classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    # classes = ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

    # TODO temporal loss dynamically change weight?
    mms = MMS(content_path, net, args.epoch_start, args.epoch_end, 512, 10, classes, cmap="tab10", resolution=100, neurons=256, verbose=1, temporal=False, split=-1)
    mms.data_preprocessing()
    # mms.prepare_visualization_for_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # PROGRAM level args
    parser.add_argument("--content_path", type=str)
    parser.add_argument("--epoch_start", type=int)
    parser.add_argument("--epoch_end", type=int)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--advance_attack", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()
    main(args)


