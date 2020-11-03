"""
calculate the time to run deepview on different number of samples
"""
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision
from deepview import DeepView
import matplotlib.pyplot as plt
import math
from cifar10_models import *
import time

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
    trainset, testset = load_CIFAR10_dataset()
    testloader = torch.utils.data.DataLoader(testset, batch_size=MAX_SAMPLES,
                                             shuffle=True, num_workers=0)

    softmax = torch.nn.Softmax(dim=-1)
    def pred_wrapper(x):
        with torch.no_grad():
            x = x.astype('float64')
            tensor = torch.from_numpy(x).to(device, dtype=torch.float)
            logits = model.fc(tensor)
            probabilities = softmax(logits).cpu().numpy()
        return probabilities

    def visualization(image, point2d, pred, label=None, title=None):
        f, a = plt.subplots()
        a.set_title(title)
        a.imshow(image.transpose([1, 2, 0]))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ---------------------deepview------------------------------
    batch_size = 100
    max_samples = MAX_SAMPLES + 2
    data_shape = (2048,)
    n = 5
    lam = 0
    resolution = 100
    cmap = 'tab10'
    title = 'ResNet-56 - CIFAR10 GAP layer-deepview inverse'

    umapParms = {
        "random_state": 42 * 42,
        "n_neighbors": 30,
        "spread": 1,
        "min_dist": 0.1,
        "a": 600,
    }

    X = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_ex_0.npy")
    Y = np.load("D:\\xianglin\\git_space\\adversarial _gen\\adversary_samples\\fgsm_labels_0.npy")

    test_num = [200, 2000, 10000]

    for num in test_num:
        x = torch.from_numpy(X[:num]).to(device, dtype=torch.float)
        y = Y[:num, 1]
        deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                            data_shape, n, lam, resolution, cmap, title=title, data_viz=visualization)
        deepview._init_mappers(None, None, umapParms)

        # get gap layer(divide to several batch)
        input_X = np.zeros([len(x), data_shape[0]])
        n_batches = max(math.ceil(len(x) / batch_size), 1)

        for b in range(n_batches):
            r1, r2 = b * batch_size, (b + 1) * batch_size
            inputs = x[r1:r2]
            with torch.no_grad():
                pred = model.gap(inputs).cpu().numpy()
                input_X[r1:r2] = pred
            print("Finished getting batch {} of gap layer...".format(b))

        t0 = time.time()
        deepview.add_samples(input_X, y)
        t1 = time.time()
        print("The time to run {} samples take {} seconds...".format(num, t1-t0))
        # deepview.savefig("result//evaluation//time//{}_{:.2}.png".format(num, t1 - t0))
