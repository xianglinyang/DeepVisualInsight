"""
find the average distance of gap layer data in cifar10 dataset
"""

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
import math
from cifar10_models import *

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

    softmax = torch.nn.Softmax(dim=-1)
    def pred_wrapper(x):
        with torch.no_grad():
            tensor = torch.from_numpy(x).to(device, dtype=torch.float)
            logits = model.fc(tensor)
            probabilities = softmax(logits).cpu().numpy()
        return probabilities

    batch_size = 200
    data_shape = (2048,)
    test_num = 500

    # the samples after normalization (10000, 3, 32, 32)
    X = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_ex_0.npy")
    Y_true = np.array(torchvision.datasets.CIFAR10(root='data', train=False, download=True).targets)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    for cls in range(10):
        index = np.where(Y_true==cls)
        index = index[:test_num]
        raw_input_X = torch.from_numpy(X[index]).to(device)

        input_X = np.zeros([len(raw_input_X), data_shape[0]])
        n_batches = max(math.ceil(len(raw_input_X) / batch_size), 1)
        for b in range(n_batches):
            r1, r2 = b * batch_size, (b + 1) * batch_size
            inputs = raw_input_X[r1:r2]
            with torch.no_grad():
                pred = model.gap(inputs).cpu().numpy()
                input_X[r1:r2] = pred

        dis = 0.0
        num = 0
        for i in range(len(input_X)):
            for j in range(i+1, len(input_X)):
                num = num+1
                dis += np.linalg.norm(input_X[i]-input_X[j])

        avg_dis = dis / num
        print("The average distance for class {} is {:.2f}".format(classes[cls], avg_dis))

