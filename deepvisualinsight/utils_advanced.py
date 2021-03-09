import numpy as np
from sklearn.cluster import KMeans
import time
import torch
import torch.nn as nn
import json
import os
import sys
import math

def clustering(data, predictions, n_clusters_per_cls, n_class=10, verbose=0):
    """
    clustering function
    :param data: [n_samples, n_features]
    :param predictions: predictions of shape [n_samples]
    :param n_clusters_per_cls: int, how many clusters that user want for each class
    :param verbose, by default 0
    :return: centers, shape [n_class, n_clusters_per_cls, n_features]
    """
    centers = []
    
    t0 = time.time()
    for i in tqdm(range(n_class)):
        c = data[np.argwhere(predictions == i).squeeze()]
        if len(c) == 0: # no data is predicted as label i
            centers.append([])
        else:
            kmeans = KMeans(n_clusters=n_clusters_per_cls, random_state=0).fit(c) # perform Kmeans clustering
            centers.append(kmeans.cluster_centers_)
    t1 = time.time()
    
    centers = np.asarray(centers)
#     print(centers.shape) #TODO: comment this 
    
    if verbose > 0:
        print("Clustering {:d} classes in {:.2f} seconds...".format(len(centers), t1 - t0))
    return centers

def cal_loss(output, target, dist, scale_const):
    '''Helper function Compute loss for C&W L2'''
    # compute the probability of the label class versus the maximum other
    real = target
    other = ((1. - target) * output - target * 10000.).max(1)[0]
    # if targeted, optimize for making the other class most likely
    loss1 = torch.clamp(other - real + confidence, min=0.)  # equiv to max(..., 0.)
    loss1 = torch.sum(scale_const * loss1)

    loss2 = dist.sum()

    loss = loss1 + loss2
    return loss


def adv_attack(image, epsilon, data_grad):
    """fgsm adversarial attack"""
    sign_data_grad = torch.sign(data_grad)
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image


def get_border_points(data_, target_, model, diff, device, epsilon=.01, limit=5,):
    """get border points by fgsm adversarial attack?"""

    num = len(data_)
    model.eval()

    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    adv_data = np.zeros(data_.shape) 
    adv_pred_labels = np.zeros(target_.shape)
    r = 0

    for i in range(num): # generate adversarial for each CP
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
            abs_dis = sort[0, 0] - sort[0, 1]

            final_pred = output.max(1, keepdim=True)[1]

            adv_ex = perturbed_data.squeeze(0).detach().cpu().numpy()
            data = torch.from_numpy(np.expand_dims(adv_ex, axis=0)).to(device)
            data.requires_grad = True
            j = j + 1
            
            # Stop when ... 
            # successfully flip the label
            if final_pred.item() != target: 
                if abs_dis < diff:
                    adv_data[r] = adv_ex
                    adv_pred_labels[r] = final_pred.item()
                    r = r + 1
                break
            # reach the boundary
            if abs_dis < diff: 
                adv_data[r] = adv_ex
                adv_pred_labels[r] = final_pred.item()
                r = r + 1
                break
             # number of iterations reach maximum iteration allowed
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


def softmax_model(model): # softmax layer
    return torch.nn.Sequential(*(list(model.children())[-1:]))


def gap_model(model): # GAP layer
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
            input_X[r1:r2] = pred.squeeze()
    return input_X

