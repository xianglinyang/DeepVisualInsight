"""
to compare lambda with gap layer data in the projection, we run lambda=0.65 as in their paper on the new metric
"""
from sklearn.neighbors import NearestNeighbors
import numpy as np
import umap
import json
import torch
import torchvision
from deepview import DeepView
import matplotlib.pyplot as plt
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
        # deepview_sim.append(len(np.intersect1d(deepview_eucli_dist[i],deepview_embedded_dist[i])) / float(n_neighbors))
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

    with open('result\evaluation\\lambda\\deepview_proj.json', 'w') as fp:
        json.dump(deepview_proj, fp)

    with open('result\evaluation\\lambda\\umap_proj.json', 'w') as fp:
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
            logits = model(tensor)
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
    data_shape = (3, 32, 32)
    n = 5
    lam = 0.65
    resolution = 100
    cmap = 'tab10'
    title = 'ResNet-56 - CIFAR10 layer (200 images)-deepview inverse'

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

    test_num = 1000

    x = X[:test_num]
    y = Y[:test_num, 1]

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title, data_viz=visualization)
    deepview._init_mappers(None, None, umapParms)

    input_X = np.array(x, copy=True)
    output_Y = np.array(y, copy=True)

    deepview.add_samples(input_X, output_Y)
    evaluate_projection(deepview, input_X)