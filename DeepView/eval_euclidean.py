"""
evaluate euclidean as an proper metric to measure the distance between high D and low D data
 we manually check the similarity between some images, their gap euclidean distance and 2D embedded points distance
 if the euclidean distance of gap layer data and 2D embedded points can represent the similarity of images, then
 we can make sure the euclidean metric is good enough to be an evaluation metric
"""
import numpy as np
import torch
import torchvision
from deepview import DeepView
import math
from cifar10_models import *
import matplotlib.pyplot as plt


CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
MAX_SAMPLES = 10000

if __name__ == "__main__":
    # ---------------------choose device------------------------------
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # ---------------------load models------------------------------
    model = resnet50(pretrained=True)
    model.eval()
    model.to(device)
    print("Load Model successfully...")

    # ---------------------load dataset------------------------------
    softmax = torch.nn.Softmax(dim=-1)

    def pred_wrapper(x):
        with torch.no_grad():
            tensor = torch.from_numpy(x).to(device, dtype=torch.float)
            logits = model.fc(tensor)
            probabilities = softmax(logits).cpu().numpy()
        return probabilities

    def visualization(image, point2d, pred, label=None, title=None):
        f, a = plt.subplots()
        a.set_title(title)
        a.imshow(image.transpose([1, 2, 0]))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                           download=True)
    testset = testset.data

    # ---------------------deepview------------------------------
    batch_size = 200
    max_samples = MAX_SAMPLES + 2
    data_shape = (2048,)
    n = 5
    lam = 0
    resolution = 100
    cmap = 'tab10'
    title = 'ResNet-56 - CIFAR10 GAP layer (200 images)-deepview inverse'

    umapParms = {
        "random_state": 42 * 42,
        "n_neighbors": 30,
        "spread": 1,
        "min_dist": 0.1,
        "a": 600,
    }

    X = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_ex_0.npy")
    Y = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_labels_0.npy")
    Y_true = np.array(torchvision.datasets.CIFAR10(root='data', train=False, download=True).targets)

    test_num = 100

    x = torch.from_numpy(X[:test_num]).to(device, dtype=torch.float)
    y = Y[:test_num, 1]

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title, data_viz=visualization)
    deepview._init_mappers(None, None, umapParms)

    raw_input_X = x.clone()
    input_X = np.zeros([len(raw_input_X), data_shape[0]])
    n_batches = max(math.ceil(len(raw_input_X) / batch_size), 1)
    for b in range(n_batches):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = raw_input_X[r1:r2]
        with torch.no_grad():
            pred = model.gap(inputs).cpu().numpy()
            input_X[r1:r2] = pred

    output_Y = np.array(y, copy=True)

    deepview.add_samples(input_X, output_Y)

    for cls in range(10):
        index = np.where(Y_true==cls)
        index = index[0][:5]
        print("class {} have index picture {}".format(cls, index))
        gaps = input_X[index]
        embedded = deepview.embedded[index]
        for i in range(5):
            plt.plot()
            plt.title("{} {}".format(classes[cls], i))
            plt.imshow(testset[index[i]])
            plt.savefig("result\\evaluation\\euclidean\\{}_{}".format(classes[cls], i))
            plt.cla()
            for j in range(i+1, 5):
                gap_dis = np.linalg.norm(gaps[i]-gaps[j])
                embedded_dis = np.linalg.norm(embedded[i]-embedded[j])
                print("({},{}) have gap dis {:.2f}, embedded dis {:.2f}".format(i, j, gap_dis, embedded_dis))
        print("-----------------------------------")
