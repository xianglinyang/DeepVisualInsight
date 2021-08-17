import numpy as np
from sklearn.cluster import KMeans
import time
import torch
import torch.nn as nn
import json
import os
import sys
import math
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F


def clustering(data, predictions, n_clusters_per_cls, n_class=10, verbose=0):
    """
    clustering function
    :param data: GAP, numpy.ndarray of shape (N, 512)
    :param predictions: class prediction, numpy.ndarray of shape (N,)
    :param n_clusters_per_cls: number of clusters for each class, int
    :param n_class: number of classes, int e.g. 10 for CIFAR10
    :param verbose: enable message printing
    :return kmeans_result: predicted cluster label, numpy.ndarray of shape (N,)
    :return predictions: same as input
    """
    kmeans_result = np.zeros_like(predictions) # save kmeans results
    
    t0 = time.time()
    for i in tqdm(range(n_class)):
        c = data[np.argwhere(predictions == i).squeeze()]
        if len(c) < n_clusters_per_cls: # no data is predicted as label i
            continue
        else:
            # perform Kmeans clustering
            kmeans = KMeans(n_clusters=n_clusters_per_cls, random_state=0).fit(c) 
            kmeans_result[np.argwhere(predictions == i).squeeze()] = kmeans.labels_ # get cluster labels
            
    t1 = time.time()
    
    if verbose > 0:
        print("Clustering {:d} classes in {:.2f} seconds...".format(n_class, t1 - t0))
        
    return kmeans_result, predictions


def cw_l2_attack(model, split, image, label, target_cls, target_gap, device, diff=0.1,
                 c=1, kappa=0, max_iter=25, learning_rate=0.1, verbose=1) :
    '''
    Implementation of C&W L2 targeted attack, Modified from https://github.com/Harry24k/CW-pytorch
    :param model: subject model
    :param image: image to attack, torch.Tensor of shape (1, C, H, W)
    :param label: original predicted class, int value 
    :param target_cls: target class int, value 
    :param target_gap: targeted gap layer, torch.Tensor of shape (1, 512)
    :param diff: threshold in deciding decision boundary
    :param c: trade-off parameter in confidence loss, float
    :param kappa: margin in confidence loss, float
    :param max_iter: maximum number of attack iter, int
    :param learning_rate: learning rate for optimizer, float
    :verbose: enable printing
    :return attack_image: perturbed image, torch.Tensor of shape (1, C, H, W)
    :return successful: indicating whether the attack is successful or not, boolean value
    '''
    
    # Get loss2
    def f(x):
        
        output = model(x)
        one_hot_label = torch.eye(len(output[0]))[label].to(device)
        one_hot_target = torch.eye(len(output[0]))[target_cls].to(device)

        # confidence for the original predicted class and target class
        i, j = torch.masked_select(output, one_hot_label.bool()), torch.masked_select(output, one_hot_target.bool())
        
        # optimize for making the other class most likely 
        return torch.clamp(i-j, min=-kappa)
    
    successful = False
    
    # initialize w : the noise
    w = torch.zeros_like(image, requires_grad=True).to(device)
    optimizer = optim.Adam([w], lr=learning_rate) # an optimizer specifically for w

    for step in range(max_iter):
        # w is the noise added to the original image, restricted to be [-1, 1]
        a = image + torch.tanh(w) 

        loss1 = nn.MSELoss(reduction='sum')(a, image) # ||x-x'||2
        loss2 = torch.sum(c*f(a)) # c*{f(label) - f(target_cls)}
        
        # Add a third loss to minimize the distance between gap layers
        gap_a = gap_model(model, split)(a)
        gap_a = gap_a.view((gap_a.shape[0], -1))
        loss3 = nn.MSELoss(reduction='sum')(gap_a, target_gap.to(device)) 

        cost = loss1 + loss2 + loss3
        
        # Backprop: jointly optimize the loss
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # New prediction
        with torch.no_grad():
            pred_new = model(a)
            conf_max = torch.max(pred_new.detach().cpu(), dim=1)[0]
            conf_min = torch.min(pred_new.detach().cpu(), dim=1)[0]
            normalized = (pred_new.detach().cpu() - conf_min)/(conf_max-conf_min) # min-max rescaling
            
        # Stop when ... 
        # successfully flip the label
        if torch.argmax(pred_new, dim=1).item() == target_cls: 
            successful = True
            break
            
        # or reach the decision boundary
        if torch.abs(normalized[0, label] - normalized[0, target_cls]).item() < diff:
            successful = True
            break 
            
        if verbose > 0:
            print('- Learning Progress : %2.2f %% ' %((step+1)/max_iter*100), end='\r')
            
    # w is the noise added to the original image, restricted to be [-1, 1]
    attack_images = image + torch.tanh(w) 

    return attack_images.detach().cpu(), successful, step


def mixup_bi(model, image1, image2, label, target_cls, device, diff=0.1, max_iter=8, alpha=0.8, verbose=1):
    '''Get BPs based on mixup method, fast
    :param model: subject model
    :param image1: images, torch.Tensor of shape (N, C, H, W)
    :param image2: images, torch.Tensor of shape (N, C, H, W)
    :param label: int, 0-9, prediction for image1
    :param target_cls: int, 0-9, prediction for image2
    :param device: the device to run code, torch cpu or torch cuda
    :param diff: the difference between top1 and top2 logits we define as boundary, float by default 0.1
    :param max_iter: binary search iteration maximum value, int by default 8
    :param verbose: whether to print verbose message, int by default 1
    '''
    def f(x):
        # New prediction
        with torch.no_grad():
            x = x.to(device, dtype=torch.float)
            pred_new = model(x)
            conf_max = torch.max(pred_new.detach().cpu(), dim=1)[0]
            conf_min = torch.min(pred_new.detach().cpu(), dim=1)[0]
            normalized = (pred_new.detach().cpu() - conf_min) / (conf_max - conf_min)  # min-max rescaling
        return pred_new, normalized

    # initialze upper and lower bound
    # set a limitation to upper bound
    upper = 1
    lower = alpha
    successful = False

    for step in range(max_iter):

        # take middle point
        lamb = (upper + lower) / 2
        image_mix = lamb * image1 + (1 - lamb) * image2
        # clip image
        image_mix = torch.clamp(image_mix, 0, 1)

        pred_new, normalized = f(image_mix)

        # Bisection method
        if normalized[0, label] - normalized[0, target_cls] > 0:  # shall decrease weight on image 1
            upper = lamb

        else:  # shall increase weight on image 1
            lower = lamb

        # Stop when ...
        # successfully flip the label
        # if torch.argmax(pred_new, dim=1).item() == target_cls:
        #     successful = True
        #     break

        # reach the decision boundary, and
        # abs(upper-lower) < 0.1 or step>1
        sorted, _ = torch.sort(normalized[0])
        curr_diff = sorted[-1] - sorted[-2]
        if curr_diff < diff and (upper-lower) < 0.1:
        # if torch.abs(normalized[0, label] - normalized[0, target_cls]).item() < diff and (upper-lower) < 0.1:
            successful = True
            break

    return image_mix, successful, step


def get_border_points_cw(model, split, input_x, gaps, confs, kmeans_result, predictions, device,
                      num_adv_eg=5000, num_cls=10, n_clusters_per_cls=10, verbose=1):
    '''Get BPs
    :param model: subject model
    :param input_x: images, torch.Tensor of shape (N, C, H, W)
    :param gaps: GAP, torch.Tensor of shape (N, 512)
    :param kmeans_result: predicted cluster label, numpy.ndarray of shape (N,)
    :param predictions: class prediction, numpy.ndarray of shape (N,)
    :param num_adv_eg: number of adversarial examples to be generated, int
    :param num_cls: number of classes, int eg 10 for CIFAR10
    :param n_clusters_per_cls: number of clusters for each class, int
    :return adv_examples: adversarial images, torch.Tensor of shape (N, C, H, W)
    :return attack_steps_ct: attacking steps
    '''

    ct = 0
    adv_examples = torch.tensor([])
    attack_steps_ct = []

    t0 = time.time()
    while ct < num_adv_eg:

        # randomly select two classes
        cls1 = np.random.choice(range(num_cls), 1)[0]
        while np.sum(predictions == cls1) <= 1: # avoid empty class
            cls1 = np.random.choice(range(num_cls), 1)[0]
        cls2 = cls1
        while cls2 == cls1 or np.sum(predictions == cls2) <= 1: # choose a different class,  avoid empty class
            cls2 = np.random.choice(range(num_cls), 1)[0]

        # randomly select one cluster from each class
        cluster1 = np.random.choice(range(n_clusters_per_cls), 1)[0]
        while np.sum((predictions == cls1) & (kmeans_result == cluster1)) <= 1: # avoid empty cluster
            cluster1 = np.random.choice(range(n_clusters_per_cls), 1)[0]

        cluster2 = np.random.choice(range(n_clusters_per_cls), 1)[0]
        while np.sum((predictions == cls2) & (kmeans_result == cluster2)) <= 1: # avoid empty cluster
            cluster2 = np.random.choice(range(n_clusters_per_cls), 1)[0]

        # randomly select one image for each cluster
        data1_index = np.argwhere((predictions == cls1) & (kmeans_result == cluster1)).squeeze()
        data2_index = np.argwhere((predictions == cls2) & (kmeans_result == cluster2)).squeeze()

        # probability to be sampled is inversely proportinal to the distance to "targeted" decision boundary
        # smaller class1-class2 is preferred
        conf1 = confs[data1_index]
        pvec1 = (1/(conf1[:,cls1]-conf1[:,cls2]+1e-4)) / torch.sum((1/(conf1[:,cls1]-conf1[:,cls2]+1e-4))) 
        conf2 = confs[data2_index]         
        pvec2 = (1/(conf2[:,cls2]-conf2[:,cls1]+1e-4)) / torch.sum((1/(conf2[:,cls2]-conf2[:,cls1]+1e-4)))
        

        image1_idx = np.random.choice(range(len(data1_index)), size=1, p=pvec1.numpy()) 
        image2_idx = np.random.choice(range(len(data2_index)), size=1, p=pvec2.numpy()) 

        image1 = input_x[data1_index[image1_idx]]
        image2 = input_x[data2_index[image2_idx]]

        gap1 = gaps[data1_index[image1_idx]]
        gap2 = gaps[data2_index[image2_idx]]

        # attack from cluster 1 to cluster 2
        attack1, successful1, attack_step1 = cw_l2_attack(model, split, image1.to(device, dtype=torch.float), cls1, cls2, gap2, device)

        # attack from cluster 2 to cluster 1
        attack2, successful2, attack_step2 = cw_l2_attack(model, split, image2.to(device, dtype=torch.float), cls2, cls1, gap1, device)

        if successful1:
            adv_examples = torch.cat((adv_examples, attack1), dim=0)
            ct += 1
            attack_steps_ct.append(attack_step1)
            
        if successful2:
            adv_examples = torch.cat((adv_examples, attack2), dim=0)
            ct += 1
            attack_steps_ct.append(attack_step2)
        
        if verbose:
            if ct % 1000 == 0:
                print('{}/{}'.format(ct, num_adv_eg))

    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1-t0))
        
    return adv_examples, attack_steps_ct


def get_border_points_mixup(model, split, input_x, gaps, confs, kmeans_result, predictions, device,
                      num_adv_eg=5000, num_cls=10, n_clusters_per_cls=10, alpha=0.8, verbose=1):
    '''Get BPs
    :param model: subject model
    :param input_x: images, torch.Tensor of shape (N, C, H, W)
    :param gaps: GAP, torch.Tensor of shape (N, 512)
    :param kmeans_result: predicted cluster label, numpy.ndarray of shape (N,)
    :param predictions: class prediction, numpy.ndarray of shape (N,)
    :param num_adv_eg: number of adversarial examples to be generated, int
    :param num_cls: number of classes, int eg 10 for CIFAR10
    :param n_clusters_per_cls: number of clusters for each class, int
    :return adv_examples: adversarial images, torch.Tensor of shape (N, C, H, W)
    :return attack_steps_ct: attacking steps
    '''

    ct = 0
    adv_examples = torch.tensor([])
    attack_steps_ct = []
    classes = []


    t0 = time.time()
    while ct < num_adv_eg:

        # randomly select two classes
        cls1 = np.random.choice(range(num_cls), 1)[0]
        while np.sum(predictions == cls1) <= 1:  # avoid empty class
            cls1 = np.random.choice(range(num_cls), 1)[0]
        cls2 = cls1
        while cls2 == cls1 or np.sum(predictions == cls2) <= 1:  # choose a different class,  avoid empty class
            cls2 = np.random.choice(range(num_cls), 1)[0]

        # randomly select one cluster from each class
        cluster1 = np.random.choice(range(n_clusters_per_cls), 1)[0]
        while np.sum((predictions == cls1) & (kmeans_result == cluster1)) <= 1:  # avoid empty cluster
            cluster1 = np.random.choice(range(n_clusters_per_cls), 1)[0]

        cluster2 = np.random.choice(range(n_clusters_per_cls), 1)[0]
        while np.sum((predictions == cls2) & (kmeans_result == cluster2)) <= 1:  # avoid empty cluster
            cluster2 = np.random.choice(range(n_clusters_per_cls), 1)[0]

        # randomly select one image for each cluster
        data1_index = np.argwhere((predictions == cls1) & (kmeans_result == cluster1)).squeeze()
        data2_index = np.argwhere((predictions == cls2) & (kmeans_result == cluster2)).squeeze()

        # probability to be sampled is inversely proportinal to the distance to "targeted" decision boundary
        # smaller class1-class2 is preferred
        conf1 = confs[data1_index]
        pvec1 = (1 / (conf1[:, cls1] - conf1[:, cls2] + 1e-4)) / torch.sum(
            (1 / (conf1[:, cls1] - conf1[:, cls2] + 1e-4)))
        conf2 = confs[data2_index]
        pvec2 = (1 / (conf2[:, cls2] - conf2[:, cls1] + 1e-4)) / torch.sum(
            (1 / (conf2[:, cls2] - conf2[:, cls1] + 1e-4)))

        image1_idx = np.random.choice(range(len(data1_index)), size=1, p=pvec1.numpy())
        image2_idx = np.random.choice(range(len(data2_index)), size=1, p=pvec2.numpy())

        image1 = input_x[data1_index[image1_idx]]
        image2 = input_x[data2_index[image2_idx]]

        attack, successful, attack_step = mixup_bi(model, image1, image2, cls1, cls2, device, alpha=alpha)

        if successful:
            adv_examples = torch.cat((adv_examples, attack), dim=0)
            ct += 1
            attack_steps_ct.append(attack_step)
            classes.append(cls1)

        if verbose:
            if ct % 1000 == 0:
                print('{}/{}'.format(ct, num_adv_eg))

    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1 - t0))

    return adv_examples, attack_steps_ct, classes


def get_border_points_exp(model, input_x, confs, predictions, device, num_adv_eg=5000, num_cls=10, alpha=0.8, verbose=1):
    '''Get BPs, 500 points per class
    :param model: subject model
    :param input_x: images, torch.Tensor of shape (N, C, H, W)
    :param predictions: class prediction, numpy.ndarray of shape (N,)
    :param num_adv_eg: number of adversarial examples to be generated, int
    :param num_cls: number of classes, int eg 10 for CIFAR10
    :return adv_examples: adversarial images, torch.Tensor of shape (N, C, H, W)
    :return attack_steps_ct: attacking steps
    '''

    adv_examples = torch.tensor([]).to(device)
    num_adv_per_class = num_adv_eg / num_cls

    curr_samples = np.zeros(int(num_cls*(num_cls-1)/2))
    tot_num = np.zeros(int(num_cls*(num_cls-1)/2))

    index_dict = dict()
    idx = 0
    for i in range(num_cls):
        for j in range(i+1, num_cls):
            index_dict[(i, j)] = idx
            idx += 1

    t0 = time.time()
    for cls in tqdm(range(num_cls)):
        num_adv = 0
        data1_index = np.argwhere(predictions == cls).squeeze()
        conf1 = confs[data1_index]
        # randomly select a targeted class
        while num_adv < num_adv_per_class:
            cls_target = np.random.choice(range(num_cls), 1)[0]
            while cls_target == cls:
                cls_target = np.random.choice(range(num_cls), 1)[0]

            data2_index = np.argwhere(predictions == cls_target).squeeze()

            # probability to be sampled is inversely proportinal to the distance to "targeted" decision boundary
            # smaller class1-class2 is preferred
            pvec1 = (1 / (conf1[:, cls] - conf1[:, cls_target] + 1e-4)) / np.sum(
                (1 / (conf1[:, cls] - conf1[:, cls_target] + 1e-4)))
            conf2 = confs[data2_index]
            pvec2 = (1 / (conf2[:, cls_target] - conf2[:, cls] + 1e-4)) / np.sum(
                (1 / (conf2[:, cls_target] - conf2[:, cls] + 1e-4)))

            image1_idx = np.random.choice(range(len(data1_index)), size=1, p=pvec1)
            image2_idx = np.random.choice(range(len(data2_index)), size=1, p=pvec2)

            image1 = input_x[data1_index[image1_idx]]
            image2 = input_x[data2_index[image2_idx]]

            attack, successful, attack_step = mixup_bi(model, image1, image2, cls, cls_target, device, alpha=alpha)

            i = min(cls, cls_target)
            j = max(cls, cls_target)
            tot_num[index_dict[(i,j)]] += 1
            if successful:
                adv_examples = torch.cat((adv_examples, attack), dim=0)
                num_adv += 1
                curr_samples[index_dict[(i, j)]] += 1

    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1 - t0))

    return adv_examples, curr_samples, tot_num


# pure random
def get_border_points_exp1(model, input_x, confs, predictions, device, num_adv_eg=5000, num_cls=10, alpha=0.8, verbose=1):
    '''Get BPs, 500 points per class
    :param model: subject model
    :param input_x: images, torch.Tensor of shape (N, C, H, W)
    :param predictions: class prediction, numpy.ndarray of shape (N,)
    :param num_adv_eg: number of adversarial examples to be generated, int
    :param num_cls: number of classes, int eg 10 for CIFAR10
    :return adv_examples: adversarial images, torch.Tensor of shape (N, C, H, W)
    :return attack_steps_ct: attacking steps
    '''

    adv_examples = torch.tensor([]).to(device)
    num_adv = 0

    curr_samples = np.zeros(int(num_cls*(num_cls-1)/2))
    tot_num = np.zeros(int(num_cls*(num_cls-1)/2))

    index_dict = dict()
    idx = 0
    for i in range(num_cls):
        for j in range(i+1, num_cls):
            index_dict[(i, j)] = idx
            idx += 1

    t0 = time.time()
    # for cls in tqdm(range(num_cls)):
    while num_adv < num_adv_eg:
        cls1 = np.random.choice(num_cls, size=1)[0]
        cls2 = np.random.choice(num_cls, size=1)[0]
        while cls2 == cls1:
            cls2 = np.random.choice(num_cls, size=1)[0]

        data1_index = np.argwhere(predictions == cls1).squeeze()
        conf1 = confs[data1_index]
        data2_index = np.argwhere(predictions == cls2).squeeze()
        conf2 = confs[data2_index]
        # randomly select a targeted class

        # probability to be sampled is inversely proportinal to the distance to "targeted" decision boundary
        # smaller class1-class2 is preferred
        pvec1 = (1 / (conf1[:, cls1] - conf1[:, cls2] + 1e-4)) / np.sum(
            (1 / (conf1[:, cls1] - conf1[:, cls2] + 1e-4)))
        conf2 = confs[data2_index]
        pvec2 = (1 / (conf2[:, cls2] - conf2[:, cls1] + 1e-4)) / np.sum(
            (1 / (conf2[:, cls2] - conf2[:, cls1] + 1e-4)))

        image1_idx = np.random.choice(range(len(data1_index)), size=1, p=pvec1)
        image2_idx = np.random.choice(range(len(data2_index)), size=1, p=pvec2)

        image1 = input_x[data1_index[image1_idx]]
        image2 = input_x[data2_index[image2_idx]]

        attack, successful, attack_step = mixup_bi(model, image1, image2, cls1, cls2, device, alpha=alpha)

        i = min(cls1, cls2)
        j = max(cls1, cls2)
        tot_num[index_dict[(i,j)]] += 1
        if successful:
            adv_examples = torch.cat((adv_examples, attack), dim=0)
            num_adv += 1
            curr_samples[index_dict[(i, j)]] += 1

    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1 - t0))

    return adv_examples, curr_samples, tot_num

# attack based on probabilities...(successful rate and current sample number)
# def get_border_points_exp2(model, input_x, confs, predictions, device, num_adv_eg=5000, num_cls=10, alpha=0.6,
#                            lambd=0.2, verbose=1):
def get_border_points(model, input_x, confs, predictions, device, num_adv_eg=5000, num_cls=10, alpha=0.6,
                           lambd=0.2, verbose=1):
    '''Get BPs, 500 points per class
    :param model: subject model
    :param input_x: images, torch.Tensor of shape (N, C, H, W)
    :param predictions: class prediction, numpy.ndarray of shape (N,)
    :param num_adv_eg: number of adversarial examples to be generated, int
    :param num_cls: number of classes, int eg 10 for CIFAR10
    :return adv_examples: adversarial images, torch.Tensor of shape (N, C, H, W)
    :return attack_steps_ct: attacking steps
    '''

    adv_examples = torch.tensor([]).to(device)
    num_adv = 0
    a = lambd

    succ_rate = np.ones(int(num_cls*(num_cls-1)/2))
    tot_num = np.zeros(int(num_cls*(num_cls-1)/2))
    curr_samples = np.zeros(int(num_cls*(num_cls-1)/2))

    # record index dictionary for query index pair
    idx = 0
    index_dict = dict()
    for i in range(num_cls):
        for j in range(i+1, num_cls):
            index_dict[idx] = (i, j)
            idx += 1

    t0 = time.time()
    while num_adv < num_adv_eg:
        idxs = np.argwhere(tot_num != 0).squeeze()
        succ = curr_samples[idxs] / tot_num[idxs]
        succ_rate[idxs] = succ

        curr_mean = np.mean(curr_samples)
        curr_rate = curr_mean - curr_samples
        curr_rate[curr_rate < 0] = 0
        if np.std(curr_rate) == 0:
            curr_rate = 1. / len(curr_rate)
        else:
            curr_rate = curr_rate / (1e-4 + np.sum(curr_rate))
        p = a*succ_rate + (1-a)*curr_rate
        p = p/(np.sum(p))

        selected = np.random.choice(range(len(curr_samples)), size=1, p=p)[0]
        (cls1, cls2) = index_dict[selected]
        data1_index = np.argwhere(predictions == cls1).squeeze()
        data2_index = np.argwhere(predictions == cls2).squeeze()
        conf1 = confs[data1_index]
        conf2 = confs[data2_index]

        # probability to be sampled is inversely proportinal to the distance to "targeted" decision boundary
        # smaller class1-class2 is preferred
        pvec1 = (1 / (conf1[:, cls1] - conf1[:, cls2] + 1e-4)) / np.sum(
            (1 / (conf1[:, cls1] - conf1[:, cls2] + 1e-4)))
        pvec2 = (1 / (conf2[:, cls2] - conf2[:, cls1] + 1e-4)) / np.sum(
            (1 / (conf2[:, cls2] - conf2[:, cls1] + 1e-4)))

        image1_idx = np.random.choice(range(len(data1_index)), size=1, p=pvec1)
        image2_idx = np.random.choice(range(len(data2_index)), size=1, p=pvec2)

        image1 = input_x[data1_index[image1_idx]]
        image2 = input_x[data2_index[image2_idx]]

        attack1, successful1, _ = mixup_bi(model, image1, image2, cls1, cls2, device, alpha=alpha)
        if successful1:
            adv_examples = torch.cat((adv_examples, attack1), dim=0)
            num_adv += 1
            curr_samples[selected] += 1
        tot_num[selected] += 1

        if num_adv < num_adv_eg:
            attack2, successful2, _ = mixup_bi(model, image2, image1, cls2, cls1, device, alpha=alpha)
            tot_num[selected] += 1
            if successful2:
                adv_examples = torch.cat((adv_examples, attack2), dim=0)
                num_adv += 1
                curr_samples[selected] += 1


    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1 - t0))

    return adv_examples, curr_samples, tot_num


def batch_run(model, split, data, device, batch_size=200):
    '''Get GAP layers and predicted labels for data
    :param model: subject model
    :param data: images torch.Tensor of shape (N, C, H, W)
    :param batch_size: batch size
    :return gaps: GAP torch.Tensor of shape (N, 512)
    :return preds: class prediction torch.Tensor of shape (N,)
    :return confs: last layer torch.Tensor of shape (N, ?)
    '''
    gaps = torch.tensor([])
    confs = torch.tensor([])
    preds = torch.tensor([])
    
    n_batches = max(math.ceil(len(data) / batch_size), 1)
    for b in tqdm(range(n_batches)):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = data[r1:r2]
        inputs = inputs.to(device, dtype=torch.float)
        
        with torch.no_grad():
            gap = gap_model(model, split)(inputs) # get GAP layers
            gap = gap.view((gap.shape[0], -1)) # flatten GAP layers
            
            conf = model(inputs)

            gaps = torch.cat((gaps, gap.detach().cpu()), dim=0)
            confs = torch.cat((confs, conf.detach().cpu()), dim=0)
            preds = torch.cat((preds, torch.argmax(conf, dim=1).detach().cpu().float()), dim=0)

    return gaps, preds, confs


###################################### I didn't change those functions ######################################
def load_labelled_data_index(filename):
    with open(filename, 'r') as f:
        index = json.load(f)

    return index


def softmax_model(model, split): # softmax layer
    return torch.nn.Sequential(*(list(model.children())[split:]))


def gap_model(model, split): # GAP layer
    return torch.nn.Sequential(*(list(model.children())[:split]))

##################################################################################################################


if __name__ == '__main__':
    
    ####### Those packages are for my own convenience can be removed #######
    from model import resnet18
    ########################################################################

    ## Load model
    print('Loading model ... ')
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    model = resnet18()
    checkpoint = torch.load('models/subject_model200.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # Load data
    print('Loading data ... ')
    input_x = torch.load('Training_data/training_dataset_data.pth')
    y = torch.load('Training_data/training_dataset_label.pth')
    
    # Get GAP and predicted labels
    print('Get GAP and predictions ... ')
    gaps, preds, confs = batch_run(model, input_x, batch_size=200)
    
    # Kmeans clustering
    print('Kmeans clustering ... ')
    kmeans_result, predictions = clustering(gaps.numpy(), preds.numpy(), n_clusters_per_cls=10)
    
    # Adversarial attacks
    print('Adv attack ... ')
    adv_examples, attack_steps_ct = get_border_points_cw(model=model, input_x=input_x, gaps=gaps, confs=confs,
                                                      kmeans_result=kmeans_result, predictions=predictions,
                                                      num_adv_eg=5000, num_cls=10, n_clusters_per_cls=10, verbose=1)
    # Save??
    torch.save(adv_examples, 'BPs.pt')
    