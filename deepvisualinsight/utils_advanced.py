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

####### Those packages are for my own convenience can be removed #######
from model import resnet18
########################################################################

def clustering(data, predictions, n_clusters_per_cls, n_class=10, verbose=0):
    """
    clustering function
    :param data: GAP numpy.ndarray of shape (N, 512)
    :param predictions: class prediction numpy.ndarray of shape (N,)
    :param n_clusters_per_cls: number of clusters for each class
    :param n_class: number of classes, e.g. 10 for CIFAR10
    :param verbose: enable message printing
    :return kmeans_result: predicted cluster label numpy.ndarray of shape (N,)
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

def cw_l2_attack(model, image, label, target_cls, target_gap,
                 c=1e-2, kappa=0, max_iter=1000, learning_rate=0.01, verbose=1) :
    '''
    Implementation of C&W L2 targeted attack, Modified from https://github.com/Harry24k/CW-pytorch
    :param model: subject model
    :param image: image to attack torch.Tensor of shape (1, C, H, W)
    :param label: original predicted class int value 
    :param target_cls: target class int value 
    :param target_gap: targeted gap layer torch.Tensor of shape (1, 512)
    :param c: trade-off parameter in loss float
    :param kappa: margin in confidence loss float
    :param max_iter: maximum number of attack iter int
    :param learning_rate: learning rate for optimizer float
    :verbose: enable printing
    :return attack_image: perturbed image torch.Tensor of shape (1, C, H, W)
    :return successful: indicating whether the attack is successful or not boolean value
    '''
    
    # Define f-function
    def f(x) :
        
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

    for step in range(max_iter) :
        # w is the noise added to the original image, restricted to be [-1, 1]
        a = image + torch.tanh(w) 

        loss1 = nn.MSELoss(reduction='sum')(a, image) # L2 norm between original image and perturbed image
        loss2 = torch.sum(c*f(a)) # confidence_diff between original and target class
        
        # Add a third loss to minimize the distance between gap layers
        gap_a = gap_model(model)(a)
        gap_a = gap_a.view((gap_a.shape[0], gap_a.shape[1]))
        loss3 = nn.MSELoss(reduction='sum')(gap_a, target_gap) 

        cost = loss1 + loss2 + loss3
        
        # Backprop: jointly optimize the loss
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # New prediction
        with torch.no_grad():
            pred_new = model(a)
        
        # Stop when ... 
        # successfully flip the label
        if torch.argmax(pred_new, dim=1).item() == target_cls: 
            successful = True
            break
            
        if verbose > 0:
            print('- Learning Progress : %2.2f %% ' %((step+1)/max_iter*100), end='\r')
            
    # w is the noise added to the original image, restricted to be [-1, 1]
    attack_images = image + torch.tanh(w) 

    return attack_images.detach().cpu(), successful

def get_border_points(model, input_x, gaps, kmeans_result, predictions, 
                      num_adv_eg = 5000, num_cls = 10, n_clusters_per_cls = 10, verbose=1):
    '''Get BPs
    :param model: subject model
    :param input_x: images torch.Tensor of shape (N, C, H, W)
    :param gaps: GAP torch.Tensor of shape (N, 512)
    :param kmeans_result: predicted cluster label numpy.ndarray of shape (N,)
    :param predictions: class prediction numpy.ndarray of shape (N,)
    :param num_adv_eg: number of adversarial examples to be generated
    :param num_cls: number of classes, eg 10 for CIFAR10
    :param n_clusters_per_cls: number of clusters for each class
    :return adv_examples: adversarial images torch.Tensor of shape (N, C, H, W)
    '''

    ct = 0
    adv_examples = torch.tensor([])

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
        image1_idx = np.random.choice(range(len(data1_index)), size=1) 
        image2_idx = np.random.choice(range(len(data2_index)), size=1) 

        image1 = input_x[data1_index[image1_idx]]
        image2 = input_x[data2_index[image2_idx]]

        gap1 = gaps[data1_index[image1_idx]]
        gap2 = gaps[data2_index[image2_idx]]

        # attack from cluster 1 to cluster 2
        attack1, successful1 = cw_l2_attack(model, image1.to(device), cls1, cls2, gap2)

        # attack from cluster 2 to cluster 1
        attack2, successful2 = cw_l2_attack(model, image2.to(device), cls2, cls1, gap1)

        if successful1:
            adv_examples = torch.cat((adv_examples, attack1), dim=0)
            ct += 1
        if successful2:
            adv_examples = torch.cat((adv_examples, attack2), dim=0)
            ct += 1
        
        if verbose:
            if ct % 1000 == 0:
                print('{}/{}'.format(ct, num_adv_eg))

    t1 = time.time()
    if verbose:
        print('Total time {:2f}'.format(t1-t0))
        
    return adv_examples


def batch_run(model, data, output_shape, batch_size=200):
    '''Get GAP layers and predicted labels for data'''
    gaps = torch.tensor([])
    preds = torch.tensor([])
    
    n_batches = max(math.ceil(len(data) / batch_size), 1)
    for b in range(n_batches):
        r1, r2 = b * batch_size, (b + 1) * batch_size
        inputs = data[r1:r2]
        inputs = inputs.to(device, dtype=torch.float)
        
        with torch.no_grad():
            pred = torch.argmax(model(inputs), dim=1)
            gap = gap_model(model)(inputs)
            gap = gap.view((gap.shape[0], gap.shape[1]))
            
            gaps = torch.cat((gaps, gap.detach().cpu()), dim=0)
            preds = torch.cat((preds, pred.detach().cpu().float()), dim=0)
         
    return gaps.numpy(), preds.numpy()





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
    
    ## Load model
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = resnet18()
    checkpoint = torch.load('subject_model.pth')
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # Load data
    input_x = torch.load('Training_data/training_dataset_data.pth')
    y = torch.load('Training_data/training_dataset_label.pth')
    
    # Get GAP and predicted labels
    gaps, preds = batch_run(model, input_x, output_shape, batch_size=200)
    
    # Kmeans clustering
    kmeans_result, predictions = clustering(gaps, preds, n_clusters_per_cls=10)
    
    # Adversarial attacks
    adv_examples = get_border_points(model, input_x, gaps, kmeans_result, predictions, 
                                     num_adv_eg = 5000, num_cls = 10, n_clusters_per_cls = 10, verbose=1)
    
    