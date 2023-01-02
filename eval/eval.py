import os.path
import sys
import argparse
import time
import json

from deepvisualinsight.MMS import MMS


def get_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameters for preparing data")

    parser.add_argument("--content_path", '-c', type=str, required=True)
    parser.add_argument("--epoch_start", type=int, required=True)
    parser.add_argument("--epoch_end", type=int, required=True)
    parser.add_argument("--epoch_period", type=int, default=1)
    parser.add_argument("--resolution", "-r", type=int, default=100)
    parser.add_argument("--embedding_dim", "-e", type=int, required=True)
    parser.add_argument("--neurons", '-n', type=int)
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "MNIST", "FashionMNIST"])
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("--output_dir", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()

    content_path = args.content_path
    epoch_start = args.epoch_start
    epoch_end = args.epoch_end
    epoch_period = args.epoch_period
    resolution = args.resolution
    embedding_dim = args.embedding_dim
    num_classes = args.num_classes
    dataset = args.dataset
    cuda = args.cuda
    split = args.split
    output_dir = args.output
    if cuda:
        attack_device = "cuda:0"
    else:
        attack_device = "cpu"
    if args.neurons is not None:
        neurons = args.neurons
    else:
        neurons = embedding_dim // 2

    # prepare hyperparameters
    sys.path.append(content_path)
    from Model.model import *
    try:
        net = resnet18()
    except:
        net = ResNet18()

    if dataset == "CIFAR10":
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    elif dataset == "MNIST":
        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    else:
        classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

    mms = MMS(content_path, net, epoch_start, epoch_end, epoch_period, embedding_dim, num_classes, classes,
              cmap="tab10", resolution=resolution, neurons=neurons, verbose=1, temporal=False, split=split,
              advance_border_gen=True, attack_device=attack_device)

    '''evaluate all properties'''
    res_dict = dict()
    k_neighbors = [15, 20, 30]
    for i in range(epoch_start, epoch_end+1, epoch_period):
        res_dict["time"] = dict()

        t0 = time.time()
        res_dict["proj_nn"] = dict()
        res_dict["proj_nn"]["train"] = dict()
        res_dict["proj_nn"]["train"][k_neighbors[0]] = mms.proj_nn_perseverance_knn_train(i, k_neighbors[0])
        res_dict["proj_nn"]["train"][k_neighbors[1]] = mms.proj_nn_perseverance_knn_train(i, k_neighbors[1])
        res_dict["proj_nn"]["train"][k_neighbors[2]] = mms.proj_nn_perseverance_knn_train(i, k_neighbors[2])
        t1 = time.time()
        res_dict["time"]["proj_nn_train"] = (t1-t0)/3

        res_dict["proj_nn"]["test"] = dict()
        res_dict["proj_nn"]["test"][k_neighbors[0]] = mms.proj_nn_perseverance_knn_test(i, k_neighbors[0])
        res_dict["proj_nn"]["test"][k_neighbors[1]] = mms.proj_nn_perseverance_knn_test(i, k_neighbors[1])
        res_dict["proj_nn"]["test"][k_neighbors[2]] = mms.proj_nn_perseverance_knn_test(i, k_neighbors[2])
        t2 = time.time()
        res_dict["time"]["proj_nn_test"] = (t2-t1)/3

        res_dict["proj_boundary"] = dict()
        res_dict["proj_boundary"]["train"] = dict()
        res_dict["proj_boundary"]["train"][k_neighbors[0]] = mms.proj_boundary_perseverance_knn_train(i, k_neighbors[0])
        res_dict["proj_boundary"]["train"][k_neighbors[1]] = mms.proj_boundary_perseverance_knn_train(i, k_neighbors[1])
        res_dict["proj_boundary"]["train"][k_neighbors[2]] = mms.proj_boundary_perseverance_knn_train(i, k_neighbors[2])
        t3 = time.time()
        res_dict["time"]["proj_boundary_train"] = (t3-t2)/3

        res_dict["proj_boundary"]["test"] = dict()
        res_dict["proj_boundary"]["test"][k_neighbors[0]] = mms.proj_boundary_perseverance_knn_test(i, k_neighbors[0])
        res_dict["proj_boundary"]["test"][k_neighbors[1]] = mms.proj_boundary_perseverance_knn_test(i, k_neighbors[1])
        res_dict["proj_boundary"]["test"][k_neighbors[2]] = mms.proj_boundary_perseverance_knn_test(i, k_neighbors[2])
        t4 = time.time()
        res_dict["time"]["proj_boundary_test"] = (t4-t3)/3

        res_dict["inv"] = dict()
        res_dict["inv"]["accu"] = dict()
        res_dict["inv"]["accu"]["train"] = mms.inv_accu_train(i)
        res_dict["inv"]["accu"]["test"] = mms.inv_accu_test(i)
        t5 = time.time()
        res_dict["time"]["inv_accu"] = t5 - t4

        res_dict["inv"]["conf_diff"] = dict()
        res_dict["inv"]["conf_diff"]["train"] = mms.inv_conf_diff_train(i)
        res_dict["inv"]["conf_diff"]["test"] = mms.inv_conf_diff_test(i)
        t6 = time.time()
        res_dict["time"]["inv_conf_diff"] = t5 - t4
    # test temporal preserverance
    res_dict["proj_t"] = dict()
    res_dict["proj_t"]["train"] = mms.proj_temporal_perseverance_train(k_neighbors[1])
    res_dict["proj_t"]["test"] = mms.proj_temporal_perseverance_test(k_neighbors[1])

    output_name = dataset+"_"+str(i)+".json"
    f = open(os.path.join(output_dir, output_name), "w")
    json.dump(res_dict, f)

