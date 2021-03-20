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
        if len(c) == 0: # no data is predicted as label i
            continue
        else:
            # perform Kmeans clustering
            kmeans = KMeans(n_clusters=n_clusters_per_cls, random_state=0).fit(c) 
            kmeans_result[np.argwhere(predictions == i).squeeze()] = kmeans.labels_ # get cluster labels
            
    t1 = time.time()
    
    if verbose > 0:
        print("Clustering {:d} classes in {:.2f} seconds...".format(n_class, t1 - t0))
        
    return kmeans_result, predictions


def mixup(model, image1, image2, label, target_cls, diff=0.1, max_iter=8, verbose=1):
    
    def f(x):
        # New prediction
        with torch.no_grad():
            x = x.to(device, dtype=torch.float)
            pred_new = model(x)
            conf_max = torch.max(pred_new.detach().cpu(), dim=1)[0]
            conf_min = torch.min(pred_new.detach().cpu(), dim=1)[0]
            normalized = (pred_new.detach().cpu() - conf_min)/(conf_max-conf_min) # min-max rescaling
        return pred_new, normalized

    # initialze upper and lower bound 
    upper = 1 
    lower = 0
    successful = False
    
    for step in range(max_iter):
        
        # take middle point
        lamb = (upper + lower)/2
        image_mix = lamb*image1 + (1-lamb)*image2
        
        pred_new, normalized = f(image_mix)
        
        # Bisection method 
        if normalized[0, label] - normalized[0, target_cls] > 0: # shall decrease weight on image 1
            upper = lamb
            
        else:   # shall increase weight on image 1
            lower = lamb
            
        # Stop when ... 
        # successfully flip the label
        if torch.argmax(pred_new, dim=1).item() == target_cls: 
            successful = True
            break
            
        # or reach the decision boundary
        if torch.abs(normalized[0, label] - normalized[0, target_cls]).item() < diff:
            successful = True
            break 

    return image_mix, successful, step

def get_border_points(model, input_x, confs, kmeans_result, predictions, 
                      num_adv_eg = 5000, num_cls = 10, n_clusters_per_cls = 10, verbose=1):
    '''Get BPs
    :param model: subject model
    :param input_x: images, torch.Tensor of shape (N, C, H, W)
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
        
        # probability to be sampled is inversely proportinal to the distance to decision boundary
#         sort1, _ = torch.sort(confs[data1_index], dim=1, descending=True)
#         pvec1 = (1/(sort1[:,0]-sort1[:,1]+1e-4)) / torch.sum(1/(sort1[:,0]-sort1[:,1]+1e-4)) # smaller top1-top2 is preferred
#         sort2, _ = torch.sort(confs[data2_index], dim=1, descending=True)
#         pvec2 = (1/(sort2[:,0]-sort2[:,1]+1e-4)) / torch.sum((1/(sort2[:,0]-sort2[:,1]+1e-4)))
    
        image1_idx = np.random.choice(range(len(data1_index)), size=1, p=pvec1.numpy()) 
        image2_idx = np.random.choice(range(len(data2_index)), size=1, p=pvec2.numpy()) 

        image1 = input_x[data1_index[image1_idx]]
        image2 = input_x[data2_index[image2_idx]]

        # attack from cluster 1 to cluster 2
        attack, successful, attack_step = mixup(model, image1, image2, cls1, cls2)

        if successful:
            adv_examples = torch.cat((adv_examples, attack), dim=0)
            ct += 1
            attack_steps_ct.append(attack_step)

        if verbose:
            if ct % 1000 == 0:
                print('{}/{}'.format(ct, num_adv_eg))

    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1-t0))
        
    return adv_examples, attack_steps_ct


def batch_run(model, data, batch_size=200):
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
            gap = gap_model(model)(inputs) # get GAP layers
            gap = gap.view((gap.shape[0], gap.shape[1])) # flatten GAP layers
            
            conf = model(inputs)

            gaps = torch.cat((gaps, gap.detach().cpu()), dim=0)
            confs = torch.cat((confs, conf.detach().cpu()), dim=0)
            preds = torch.cat((preds, torch.argmax(conf, dim=1).detach().cpu().float()), dim=0)

    return gaps, preds, confs


###################################### I didn't change those functions ######################################
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
    adv_examples, attack_steps_ct = get_border_points(model = model, input_x = input_x, confs = confs, kmeans_result = kmeans_result, predictions = predictions, num_adv_eg = 5000, num_cls = 10, n_clusters_per_cls = 10, verbose = 1)
    
    # Save??
    torch.save(adv_examples, 'BPs.pt')
    