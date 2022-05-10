import numpy as np
import sys
import argparse
import os
from scipy import stats
sys.path.append("/home/xianglin/projects/git_space/DeepVisualInsight/deepvisualinsight")
from MMS import MMS


def get_arguments():
    parser = argparse.ArgumentParser(description="Hyperparameters for preparing data")

    parser.add_argument("--content_path", '-c', type=str, required=True)
    parser.add_argument("--epoch_start", type=int, required=True)
    parser.add_argument("--epoch_end", type=int, required=True)
    parser.add_argument("--epoch_period", type=int, default=1)
    parser.add_argument("--resolution", "-r", type=int, default=100)
    parser.add_argument("--embedding_dim", "-e", type=int, required=True)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("-t", type=float)
    parser.add_argument("-a", type=float)
    parser.add_argument("--temporal", type=int, choices=[1, 2, 3])
    parser.add_argument("--parametricUmap", type=int, choices=[0, 1], default=0, help="whether to run baseline parametric...")
    parser.add_argument("--attention", type=int, choices=[0, 1], default=1, help="whether to add attention to renconstruction loss")
    parser.add_argument("--preprocess", type=int, choices=[0, 1], help="with 0 being false and 1 being true")

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
    cuda = args.cuda
    split = args.split
    alpha = args.a
    temperature = args.t
    temporal_strategy = args.temporal
    parametricUmap = args.parametricUmap
    attention = args.attention
    preprocess = args.preprocess

    visible_device = "3"
    # tensorflow
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_device
    # pytorch
    if cuda:
        attack_device = "cuda:{}".format(0)
    else:
        attack_device = "cpu"

    # prepare hyperparameters
    sys.path.append(content_path)
    if embedding_dim == 512:
        from Model.model import resnet18
        net = resnet18()
    else:
        from Model.model import resnet50
        net = resnet50()

    # dummy classes labels
    classes = range(10)
    eval_name = ""
    if parametricUmap:
        temporal = False
        transfer_learning = True
        eval_name += "_parametricUmap"

    if temporal_strategy == 1:
        temporal = False
        transfer_learning = False
        eval_name += "_NT"
    elif temporal_strategy == 2:
        temporal = False
        transfer_learning = True
        eval_name += "_T"
    else:
        temporal = True
        transfer_learning = True
        eval_name += "_step2"
    
    if attention:
        eval_name+="_A"

    mms = MMS(content_path, net, epoch_start, epoch_end, epoch_period, embedding_dim, num_classes, classes,
              temperature=temperature, attention=attention,
              cmap="tab10", resolution=resolution, verbose=1,
              temporal=temporal, transfer_learning=transfer_learning, step3=False,
              split=split, alpha=alpha, withoutB=parametricUmap, attack_device=attack_device)
    
    train_num = mms.train_num(1)
    test_num = mms.test_num(1)
    EPOCH = epoch_end
    # train 
    LEN = train_num
    high_repr = np.zeros((EPOCH,LEN,512))
    low_repr = np.zeros((EPOCH,LEN,2))
    for i in range(EPOCH):
        high_repr[i] = mms.get_epoch_train_repr_data(i+1)
        low_repr[i] = mms.batch_project(high_repr[i], i+1)

    epochs = [i for i in range(EPOCH)]
    corrs = np.zeros((EPOCH,LEN))
    ps = np.zeros((EPOCH,LEN))
    for i in range(LEN):
        high_embeddings = high_repr[:,i,:].squeeze()
        low_embeddings = low_repr[:,i,:].squeeze()
        for e in epochs:
            high_dists = np.linalg.norm(high_embeddings - high_embeddings[e], axis=1)
            low_dists = np.linalg.norm(low_embeddings - low_embeddings[e], axis=1)
            corr, p = stats.spearmanr(high_dists, low_dists)
            corrs[e][i] = corr
            ps[e][i] = p
    
    np.save(os.path.join(content_path, "Model", "DVI_train_corrs.npy"), corrs)
    np.save(os.path.join(content_path, "Model", "DVI_train_ps.npy"), ps)

    # test
    LEN = test_num
    high_repr = np.zeros((EPOCH,LEN,512))
    low_repr = np.zeros((EPOCH,LEN,2))
    for i in range(EPOCH):
        high_repr[i] = mms.get_epoch_test_repr_data(i+1)
        low_repr[i] = mms.batch_project(high_repr[i], i+1)

    epochs = [i for i in range(EPOCH)]
    corrs = np.zeros((EPOCH,LEN))
    ps = np.zeros((EPOCH,LEN))
    for i in range(LEN):
        high_embeddings = high_repr[:,i,:].squeeze()
        low_embeddings = low_repr[:,i,:].squeeze()
        for e in epochs:
            high_dists = np.linalg.norm(high_embeddings - high_embeddings[e], axis=1)
            low_dists = np.linalg.norm(low_embeddings - low_embeddings[e], axis=1)
            corr, p = stats.spearmanr(high_dists, low_dists)
            corrs[e][i] = corr
            ps[e][i] = p
    
    np.save(os.path.join(content_path, "Model", "DVI_test_corrs.npy"), corrs)
    np.save(os.path.join(content_path, "Model", "DVI_test_ps.npy"), ps)
