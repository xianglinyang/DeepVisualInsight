import torch
import math
import time
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from cifar10_models import *

def save_checkpoint(state,filename='checkpoint.pth'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    total = 0
    correct = 0

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input = input.to(device)
        target_var = target

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        _, predicted = output.max(1)
        losses.update(loss.item(), input.size(0))
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 50 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc:.3f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, acc=correct / total))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# load model
model = resnet50(pretrained=False)
model.load_state_dict(torch.load('parametric_umap_models/resnet50_data_augmentation/model3.th')['state_dict'])

print("Load Model successfully...")

import torchvision.transforms as transforms
# Data augmentation and normalization for training
# Just normalization for validation
CIFAR_NORM = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomCrop(size=[32, 32], padding=3),
        # transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        # transforms.RandomPerspective(),
        # transforms.RandomRotation(180),
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR_NORM)
    ]),
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR_NORM)
    ]),
}

trainset = torchvision.datasets.CIFAR10(root='data', train=True,
                                        download=True, transform=data_transforms['train'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                          shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='data', train=False,
                                       download=True, transform=data_transforms['test'])
testloader = torch.utils.data.DataLoader(testset, batch_size=2000,
                                         shuffle=False, num_workers=2)

classes = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# Initialize the network
# model = model.to(device)

# freeze front layers
ct = 0
for child in model.children():
    ct += 1
    if ct < 5:
        for param in child.parameters():
            param.requires_grad = False
model = model.to(device)
# model.eval()

import torch.nn as nn
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=5)
criterion = nn.CrossEntropyLoss()

global best_prec1
best_prec1 = 0

for epoch in range(5):
    model.train()

    # train for one epoch
    print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
    train(trainloader, model, criterion, optimizer, epoch)
    lr_scheduler.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 testing images:{:.2f}%'.format(100 * correct / total))


    if epoch > 0 and epoch % 50 == 0:
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict()
        },  filename=os.path.join('parametric_umap_models/resnet50_data_augmentation', 'checkpoint3.th'))

    save_checkpoint({
        'state_dict': model.state_dict()
    },  filename=os.path.join('parametric_umap_models/resnet50_data_augmentation', 'model3.th'))
