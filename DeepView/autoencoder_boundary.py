import numpy as np
import json
import torch
import torchvision
from deepview import DeepView
import matplotlib.pyplot as plt
import math
from cifar10_models import *
import os
import tensorflow as tf
import torchvision.transforms as transforms

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

save_location = "parametric_umap_models\parametric_umap_autoencoder"
# load encoder
encoder_output = os.path.join(save_location, "encoder")
if os.path.exists(encoder_output):
    encoder = tf.keras.models.load_model(encoder_output)
    print("Keras encoder model loaded from {}".format(encoder_output))

# load decoder
decoder_output = os.path.join(save_location, "decoder")
if os.path.exists(decoder_output):
    decoder = tf.keras.models.load_model(decoder_output)
    print("Keras decoder model loaded from {}".format(decoder_output))

CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
MAX_SAMPLES = 10000

# ---------------------choose device------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
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
        logits = model.fc(tensor).cpu().numpy()
        # probabilities = softmax(logits).cpu().numpy()
    return logits

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

CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(*CIFAR_NORM)])

# device = torch.device("cpu")
testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200,
                                          shuffle=True, num_workers=0)
testing_data = np.zeros((10000, 3, 32, 32))
for i, (data, target) in enumerate(testloader, 0):
    r1, r2 = i * 200, (i + 1) * 200
    testing_data[r1:r2] = data

raw_input_X = torch.from_numpy(testing_data).to(device, dtype=torch.float)
input_X = np.zeros([len(raw_input_X), data_shape[0]])
output_Y = np.zeros(len(raw_input_X))
n_batches = max(math.ceil(len(raw_input_X) / batch_size), 1)
for b in range(n_batches):
    r1, r2 = b * batch_size, (b + 1) * batch_size
    inputs = raw_input_X[r1:r2]
    with torch.no_grad():
        pred = model.gap(inputs).cpu().numpy()
        input_X[r1:r2] = pred
        pred = pred_wrapper(pred).argmax(axis=1)
        output_Y[r1:r2] = pred


test_num = 10000

deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                    data_shape, n, lam, resolution, cmap, title=title, data_viz=visualization,
                    clip_certainty=1.5, metric="parametricUmap", encoder=encoder, decoder=decoder)


deepview.add_samples(input_X[:test_num], output_Y[:test_num])
deepview.savefig("result\\evaluation\\parametricUmap\\inverse.png")