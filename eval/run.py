import sys
import argparse

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
    parser.add_argument("-t", type=float)
    parser.add_argument("-a", type=float)

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
    alpha = args.a
    temperature = args.t
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
        net = resnet50()
    except:
        net = ResNet18()
        # net = resnet50()
    # net = CIFAR_17()

    if dataset == "CIFAR10":
        classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
    elif dataset == "MNIST":
        classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")
    else:
        classes = ("T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")

    mms = MMS(content_path, net, epoch_start, epoch_end, epoch_period, embedding_dim, num_classes, classes, temperature=temperature,
              cmap="tab10",resolution=resolution, neurons=neurons, verbose=1, temporal=False, split=split,
              advance_border_gen=True, alpha=alpha, withoutB=False, attack_device=attack_device)

    # encoder_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(136), "encoder_advance")
    # encoder = tf.keras.models.load_model(encoder_location)
    # decoder_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(136), "decoder_advance")
    # decoder = tf.keras.models.load_model(decoder_location)

    mms.data_preprocessing()
    mms.prepare_visualization_for_all()
    mms.save_evaluation(eval=True)