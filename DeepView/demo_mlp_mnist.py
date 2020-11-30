"""
This is the demo for how to use other distance metric in deepview instead of "precomputed" kernel matrix
The main differences are:
    add metric="euclidean" to config parameter
    add metric="euclidean" when initializing deepview
    add clip_certainty=0.25 when initializing deepview
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from deepview import DeepView
import json

def mnist(cuda=True, model_root=None):
    print("Building and initializing mnist parameters")
    from mnist_models.mnist_model import mnist
    m = mnist(pretrained=True)
    if cuda:
        m = m.cuda()
    return m

with open('DeepView\\mnist_data.json', 'r') as f:
    dict = json.load(f)

X = []
Y = []

for data in dict["sampled_data"]:
    vector = []
    for dimension in data["vector"]:
        vector.append(data["vector"][dimension])
    X.append(vector)
    Y.append(data["metadata"]["label"])

X = X[:500]
Y = Y[:500]

X = np.array(X)
Y = np.array(Y)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch_model = mnist(cuda=torch.cuda.is_available())
softmax = torch.nn.Softmax(dim=-1)


# this is the prediction wrapper, it encapsulates the call to the model
# and does all the casting to the appropriate datatypes
def pred_wrapper(x):
    with torch.no_grad():
        x = np.array(x, dtype=np.float32)
        tensor = torch.from_numpy(x).to(device)
        logits = torch_model(tensor)
        probabilities = softmax(logits).cpu().numpy()
    return probabilities


def visualization(image, point2d, pred, label=None, title=None):
    f, a = plt.subplots()
    a.set_title(title)
    a.imshow(image.transpose([1, 2, 0]))


# the classes in the dataset to be used as labels in the plots
classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
# --- Deep View Parameters ----
batch_size = 512
max_samples = 5000
data_shape = (784,)
lam = 0  # default parameter
title = 'MLP - MNIST'

umapParms = {
    "random_state": 42 * 42,
    "n_neighbors": 30,
    "spread": 1,
    "min_dist": 0.1,
    "a": 600,
    "metric": "euclidean"
}

deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                    data_shape, lam=lam, title=title, metric="euclidean", clip_certainty=0.25)
deepview._init_mappers(None, None, umapParms)

deepview.add_samples(X, Y)
deepview.show()

