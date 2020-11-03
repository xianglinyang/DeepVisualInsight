"""
evaluate the projection function in deepview:
    do knn to every point from high D and low D, find the proportion of their sharing neighbors
    example:
        k=4, by knn we know
        high D has neighbor (1,3,5,7)
        low D has neighbor (2,3,5,7)
        the outcome would be 3/4 = 0.75
in the mean time, we compare deepview projection with umap projection
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import umap
import json
import torch
import torchvision
from deepview import DeepView
import matplotlib.pyplot as plt
import math
from cifar10_models import *


CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
MAX_SAMPLES = 10000

def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

def knn(deepview, umap_unsup, X, n_neighbors=5):

    deepview_sim = list()
    umap_sim = list()

    embedding_sup = deepview.embedded
    dists = deepview.distances
    embedding_unsup = umap_unsup.fit_transform(X)

    # deepview embedded dist
    deepview_embedded = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    deepview_embedded.fit(embedding_sup)
    deepview_embedded_dist = deepview_embedded.kneighbors(return_distance=False)

    eucli = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    eucli.fit(X)
    eucli_dist = eucli.kneighbors(return_distance=False)

    umap_embedded = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    umap_embedded.fit(embedding_unsup)
    umap_embedded_dist = umap_embedded.kneighbors(return_distance=False)

    for i in range(len(X)):
        deepview_sim.append(len(np.intersect1d(eucli_dist[i], deepview_embedded_dist[i])) / float(n_neighbors))
        umap_sim.append(len(np.intersect1d(eucli_dist[i], umap_embedded_dist[i])) / float(n_neighbors))

    return deepview_sim, umap_sim


def evaluate_projection(deepview, X):
    if len(np.shape(X)) > 2:
        bs = len(X)
        X = X.reshape(bs, -1)

    neighbors = 30
    umap_unsup = umap.UMAP(n_neighbors=neighbors, random_state=11 * 12 * 13)

    deepview_proj = dict()
    umap_proj = dict()

    test_prop = [0.01, 0.05, 0.1]
    test_num = [int(x * len(X)) for x in test_prop]

    for k in test_num:
        deepview_sim, umap_sim = knn(deepview, umap_unsup, X, n_neighbors=k)
        deepview_proj[k] = float(sum(deepview_sim)) / len(deepview_sim)
        umap_proj[k] = float(sum(umap_sim)) / len(umap_sim)
        print("finish finding the {}-th neighbors...".format(k))

    with open('result\\evaluation\\projection\\deepview_proj.json', 'w') as fp:
        json.dump(deepview_proj, fp)

    with open('result\\evaluation\\projection\\umap_proj.json', 'w') as fp:
        json.dump(umap_proj, fp)

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

    test_num = 200

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
    evaluate_projection(deepview, input_X)