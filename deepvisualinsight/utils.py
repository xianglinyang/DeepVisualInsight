import numpy as np
from sklearn.cluster import KMeans
import time
import torch
import torch.nn as nn
import json
import os
import sys
import math


def clustering(data, n_clusters, verbose=0):
    """
    clustering function
    :param data: [n_samples, n_features]
    :param n_clusters: int, how many clusters that user want
    :param verbose, by default 0
    :return: centers, [n_clusters, n_features]
    """
    data_shape = data.shape
    center_shape = (n_clusters,) + data_shape[1:]

    if data_shape[0] <= 10000:
        t0 = time.time()
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
        t1 = time.time()
        if verbose > 0:
            print("Clustering {:d} classes in {:.2f} seconds...".format(n_clusters, t1 - t0))
        return kmeans.cluster_centers_

    t0 = time.time()
    kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
    t1 = time.time()
    if verbose > 0:
        print("Clustering 10 classes in {:.2f} seconds...".format(t1 - t0))

    centers = np.zeros(shape=center_shape)
    labels = kmeans.labels_
    t0 = time.time()
    r1 = 0
    r2 = 0
    ratio = len(data) / n_clusters
    for i in range(10):
        index = np.argwhere(labels == i).squeeze()
        c = data[index]
        if i < 9:
            num = math.ceil(len(c) / ratio)
            r2 = r1 + num
        else:
            num = len(centers) - r1
            r2 = r1 + num
        kmeans = KMeans(n_clusters=num, random_state=0).fit(c)
        centers[r1:r2] = kmeans.cluster_centers_
        r1 = r2
    t1 = time.time()
    if verbose > 0:
        print("Clustering {:d} classes in {:.2f} seconds...".format(len(centers), t1 - t0))
    return centers


def adv_attack(image, epsilon, data_grad):
    """fgsm adversarial attack"""
    sign_data_grad = torch.sign(data_grad)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def get_border_points(data_, target_, model, device, epsilon=.01, limit=5,):
    """get border points by fgsm adversarial attack"""

    num = len(data_)
    model.eval()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    adv_data = np.zeros(data_.shape)
    adv_pred_labels = np.zeros(target_.shape)
    r = 0

    for i in range(num):
        data, target = data_[i:i+1], target_[i:i+1]
        data = data.to(device=device, dtype=torch.float)
        target = target.to(device=device, dtype=torch.long)

        data.requires_grad = True
        j = 1
        while True:
            output = model(data)

            loss = criterion(output, target)  # loss for ground-truth class
            model.zero_grad()
            loss.backward()
            data_grad = data.grad.data

            perturbed_data = adv_attack(data, epsilon, data_grad)

            output = model(perturbed_data)
            sort, _ = torch.sort(output, dim=-1, descending=True)
            abs_dis = (sort[0, 0] - sort[0, 1])/(sort[0, 0] - sort[0, -1])

            final_pred = output.max(1, keepdim=True)[1]

            adv_ex = perturbed_data.squeeze(0).detach().cpu().numpy()
            data = torch.from_numpy(np.expand_dims(adv_ex, axis=0)).to(device)
            data.requires_grad = True
            j = j + 1
            if final_pred.item() != target:
                if abs_dis < 0.1:
                    adv_data[r] = adv_ex
                    adv_pred_labels[r] = final_pred.item()
                    r = r + 1
                break
            if abs_dis < 0.1:
                adv_data[r] = adv_ex
                adv_pred_labels[r] = final_pred.item()
                r = r + 1
                break
            if j > limit:
                break
    adv_data = adv_data[:r]
    # adv_pred_labels = adv_pred_labels[:r]
    return adv_data


def load_labelled_data_index(filename):
    if not os.path.exists(filename):
        sys.exit("data file doesn't exist!")
    with open(filename, 'r') as f:
        index = json.load(f)
    return index


def softmax_model(model):
    return torch.nn.Sequential(*(list(model.children())[-1:]))


def gap_model(model):
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def batch_run(model, data, output_shape, batch_size=200):
    data = data.to(dtype=torch.float)
    input_X = np.zeros([len(data), output_shape])
    n_batches = max(math.ceil(len(data) / batch_size), 1)
    for b in range(n_batches):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = data[r1:r2]
        with torch.no_grad():
            pred = model(inputs).cpu().numpy()
            input_X[r1:r2] = pred.reshape(pred.shape[0], -1)
    return input_X
