from cifar10_models import *
import numpy as np
import torchvision
"""
the average distance of each class in cifar10 dataset after normalization...
"""

test_num = 500

X = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_ex_0.npy")
Y = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_labels_0.npy")
Y_true = np.array(torchvision.datasets.CIFAR10(root='data', train=False, download=True).targets)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

for cls in range(10):
    index = np.where(Y_true == cls)
    index = index[:test_num]
    input_X = X[index]

    dis = 0.0
    num = 0
    for i in range(len(input_X)):
        for j in range(i+1, len(input_X)):
            num = num+1
            dis += np.linalg.norm(input_X[i]-input_X[j])

    avg_dis = dis / num
    print("The average distance for class {} is {:.2f}".format(classes[cls], avg_dis))