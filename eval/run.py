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
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--split", type=int, default=-1)
    parser.add_argument("-t", type=float)
    parser.add_argument("-a", type=float)
    parser.add_argument("--temporal", type=int, choices=[1, 2, 3])
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
    preprocess = args.preprocess

    if cuda:
        attack_device = "cuda:0"
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
    if temporal_strategy == 1:
        temporal = False
        transfer_learning = False
        eval_name = "_NT"
    elif temporal_strategy == 2:
        temporal = False
        transfer_learning = True
        eval_name = "_T"
    else:
        temporal = True
        transfer_learning = True
        eval_name = "_step2"


    mms = MMS(content_path, net, epoch_start, epoch_end, epoch_period, embedding_dim, num_classes, classes,
              temperature=temperature,
              cmap="tab10", resolution=resolution, verbose=1,
              temporal=temporal, transfer_learning=transfer_learning, step3=False,
              split=split, advance_border_gen=True, alpha=alpha, withoutB=False, attack_device=attack_device)

    # encoder_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(136), "encoder_advance")
    # encoder = tf.keras.models.load_model(encoder_location)
    # decoder_location = os.path.join(content_path, "Model", "Epoch_{:d}".format(136), "decoder_advance")
    # decoder = tf.keras.models.load_model(decoder_location)

    # if preprocess == 1:
    #     mms.data_preprocessing()
    # mms.prepare_visualization_for_all()
    # mms.save_evaluation(eval=True, name=eval_name)
    mms.proj_temporal_perseverance_train(10, eval_name)
    mms.proj_temporal_perseverance_test(10, eval_name)
    mms.proj_temporal_perseverance_train(20, eval_name)
    mms.proj_temporal_perseverance_test(20, eval_name)