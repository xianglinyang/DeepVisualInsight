"""
evaluate the inverse function with lambda=.65 with three new metric
"""
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from deepview import DeepView
import matplotlib.pyplot as plt
import math
from cifar10_models import *
from deepview.embeddings import init_inv_umap
import pandas as pd


CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
MAX_SAMPLES = 10000

def load_CIFAR10_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR_NORM)])
    training_dataset = torchvision.datasets.CIFAR10(root='data', train=True,
                               download=True, transform=transform)
    testing_dataset = torchvision.datasets.CIFAR10(root='data', train=False,
                                                    download=True, transform=transform)
    return training_dataset, testing_dataset

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
    title = 'ResNet-56 - CIFAR10 GAP layer (200 images)-deepview inverse'

    umapParms = {
        "random_state": 42 * 42,
        "n_neighbors": 30,
        "spread": 1,
        "min_dist": 0.1,
        "a": 600,
    }
    test_num = 1000
    adv_succ_num = 50
    adv_fail_num = 50
    true_num = 50

    X = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_ex_0.npy")
    Y = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_labels_0.npy")
    Y_true = np.array(torchvision.datasets.CIFAR10(root='data', train=False, download=True).targets)

    x = torch.from_numpy(X[:test_num]).to(device, dtype=torch.float)
    y = Y[:test_num, 1]

    Adv_succ = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\adv_testset\\succ.npy")
    Adv_succ_labels = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\adv_testset\\succ_label.npy")
    Adv_fail = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\adv_testset\\fail.npy")
    Adv_fail_labels = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\adv_testset\\fail_label.npy")

    adv_succ = torch.from_numpy(Adv_succ[-adv_succ_num:]).to(device, dtype=torch.float)
    adv_succ_labels = Adv_succ_labels[-adv_succ_num:, 1]
    adv_fail = torch.from_numpy(Adv_fail[-adv_fail_num:]).to(device, dtype=torch.float)
    adv_fail_labels = Adv_fail_labels[-adv_fail_num:, 1]

    true_samples = torch.from_numpy(X[test_num: test_num + true_num]).to(device, dtype=torch.float)
    true_labels = Y[test_num: test_num + true_num, 1]

    ori_l = []
    adv_l = []
    new_ori_l = []
    new_adv_l = []
    confidence_ori = []
    confidence_adv = []
    new_confidence_ori = []
    new_confidence_adv = []
    distance_ori = []
    distance_adv = []

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title, data_viz=visualization)
    deepview._init_mappers(None, None, umapParms)

    raw_input_X = x.clone()
    raw_input_X = torch.cat((raw_input_X, adv_succ), axis=0)
    raw_input_X = torch.cat((raw_input_X, adv_fail), axis=0)
    raw_input_X = torch.cat((raw_input_X, true_samples), axis=0)
    input_X = raw_input_X.cpu().numpy()

    output_Y = np.array(y, copy=True)
    output_Y = np.concatenate((output_Y, adv_succ_labels), axis=0)
    output_Y = np.concatenate((output_Y, adv_fail_labels), axis=0)
    output_Y = np.concatenate((output_Y, true_labels), axis=0)

    deepview.add_samples(input_X, output_Y)

    # pick samples for training and testing
    train_samples = deepview.samples[:test_num]
    train_embeded = deepview.embedded[:test_num]
    train_labels = deepview.y_pred[:test_num]
    test_adv_succ = deepview.samples[test_num:test_num+adv_succ_num]
    test_adv_succ_embedded = deepview.embedded[test_num:test_num+adv_succ_num]
    test_adv_succ_labels = deepview.y_pred[test_num:test_num+adv_succ_num]
    test_adv_fail = deepview.samples[test_num + adv_succ_num:test_num + adv_succ_num+adv_fail_num]
    test_adv_fail_embedded = deepview.embedded[test_num + adv_succ_num:test_num + adv_succ_num+adv_fail_num]
    test_adv_fail_labels = deepview.y_pred[test_num + adv_succ_num:test_num + adv_succ_num+adv_fail_num]
    test_true = deepview.samples[-true_num:]
    test_true_embedded = deepview.embedded[-true_num:]
    test_true_labels = deepview.y_pred[-true_num:]


    # get DeepView an untrained inverse mapper
    # and train it on the train set
    deepview.inverse = init_inv_umap()
    deepview.inverse.fit(train_embeded, train_samples)

    # apply inverse mapping to embedded samples and
    # predict the reconstructions
    train_recon = deepview.inverse(train_embeded)
    train_preds = deepview.model(train_recon).argmax(-1)
    train_confidence = deepview.model(train_samples).max(axis=1)
    train_new_confidence = deepview.model(train_recon)[np.arange(len(train_preds)),train_preds]

    adv_succ_recon = deepview.inverse(test_adv_succ_embedded)
    adv_succ_preds = deepview.model(adv_succ_recon).argmax(-1)
    adv_succ_confidence = deepview.model(test_adv_succ).max(axis=1)
    adv_succ_new_confidence = deepview.model(adv_succ_recon)[np.arange(len(adv_succ_preds)),adv_succ_preds]

    adv_fail_recon = deepview.inverse(test_adv_fail_embedded)
    adv_fail_preds = deepview.model(adv_fail_recon).argmax(-1)
    adv_fail_confidence = deepview.model(test_adv_fail).max(axis=1)
    adv_fail_new_confidence = deepview.model(adv_fail_recon)[np.arange(len(adv_fail_preds)),adv_fail_preds]

    true_recon = deepview.inverse(test_true_embedded)
    true_preds = deepview.model(true_recon).argmax(-1)
    true_confidence = deepview.model(test_true).max(axis=1)
    true_new_confidence = deepview.model(true_recon)[np.arange(len(true_preds)),true_preds]


    # calculate pred accuracy
    n_correct = np.sum(train_labels == train_preds)
    train_acc = 100 * n_correct / test_num

    n_correct = np.sum(test_adv_succ_labels == adv_succ_preds)
    test_adv_succ_acc = 100 * n_correct / float(adv_succ_num)

    n_correct = np.sum(test_adv_fail_labels == adv_fail_preds)
    test_adv_fail_acc = 100 * n_correct / float(adv_fail_num)

    n_correct = np.sum(test_true_labels == true_preds)
    test_true_acc = 100 * n_correct / float(true_num)

    # calculate distance
    train_distance = np.mean(np.sqrt(np.sum(np.square(train_samples-train_recon), axis=(1,2,3))))
    adv_succ_distance = np.mean(np.sqrt(np.sum(np.square(test_adv_succ - adv_succ_recon), axis=(1,2,3))))
    adv_fail_distance = np.mean(np.sqrt(np.sum(np.square(test_adv_fail - adv_fail_recon), axis=(1,2,3))))
    true_distance = np.mean(np.sqrt(np.sum(np.square(test_true - true_recon), axis=(1,2,3))))

    # calculate confidence difference
    train_conf_diff = np.mean(np.abs(train_confidence - train_new_confidence))
    adv_succ_conf_diff = np.mean(np.abs(adv_succ_confidence - adv_succ_new_confidence))
    adv_fail_conf_diff = np.mean(np.abs(adv_fail_confidence - adv_fail_new_confidence))
    true_conf_diff = np.mean(np.abs(true_confidence - true_new_confidence))

    print("train acc:{:.2f}%, test_succ acc:{:.2f}%, test_fail acc:{:.2f}%, test_true acc:{:.2f}%".format(train_acc,
                                                                                                          test_adv_succ_acc,
                                                                                                          test_adv_fail_acc,
                                                                                                          test_true_acc))

    print("train distance:{:.2f}, test_succ dis:{:.2f}, test_fail dis:{:.2f}, test_true dis:{:.2f}".format(train_distance,
                                                                                                          adv_succ_distance,
                                                                                                          adv_fail_distance,
                                                                                                          true_distance))
    print("train con diff:{:.2f}, test_succ con dif:{:.2f}, test_fail con diff:{:.2f}, test_true con diff:{:.2f}".format(train_conf_diff,
                                                                                                          adv_succ_conf_diff,
                                                                                                          adv_fail_conf_diff,
                                                                                                          true_conf_diff))




