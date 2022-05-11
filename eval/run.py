from math import e
import sys
import argparse
import os
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

    # encoder_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(136), "encoder_advance")
    # encoder = tf.keras.models.load_model(encoder_location)
    # decoder_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(136), "decoder_advance")
    # decoder = tf.keras.models.load_model(decoder_location)

    # if preprocess == 1:
    #     mms.data_preprocessing()
    # mms.prepare_visualization_for_all()

    dataset = content_path.split("/")[-1]
    EVAL_EPOCH_DICT = {
        "resnet18_mnist":[4, 12, 20],
        "resnet18_fmnist":[10,30,50],
        "resnet18_cifar10":[40, 120, 200]
    }
    eval_epochs = EVAL_EPOCH_DICT[dataset]

    mms.save_epoch_evaluation(eval_epochs[0], eval=True, save_corr=True, eval_temporal=True, name=eval_name)
    mms.save_epoch_evaluation(eval_epochs[1], eval=True, save_corr=False, eval_temporal=False, name=eval_name)
    mms.save_epoch_evaluation(eval_epochs[2], eval=True, save_corr=False, eval_temporal=False, name=eval_name)

    