# --------- To load demo data --------------
from torchvision import datasets, transforms

# --------- Torch libs ---------------------
from models.torch_model import TorchModel
import models.resnet as resnet
import torch
import torch.nn.functional as F
import torch.nn as nn

# --------- Tensorflow ---------------------
import tensorflow as tf

# --------- SciPy libs ---------------------
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# --------- Utility libs -------------------
import matplotlib.pyplot as plt
import numpy as np


TORCH_WEIGHTS = "models/pytorch_resnet_cifar10-master/pretrained_models/resnet20-12fca82f.th"
CIFAR_NORM = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

def make_FashionMNIST_dataset():
	transform = transforms.Compose([transforms.ToTensor()])
	trainloader = torch.utils.data.DataLoader(
		datasets.FashionMNIST('data', train=True, download=True,
			transform=transforms.Compose([
				transforms.ToTensor()])),
		batch_size = 64, shuffle=False)

	testset = datasets.FashionMNIST('data', train=False,
		download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=1000,
		shuffle=True, num_workers=2)
	return trainloader, testset, testloader


def make_cifar_dataset():
	transform = transforms.Compose([
		transforms.ToTensor(),
     	transforms.Normalize(*CIFAR_NORM)])
	testset = datasets.CIFAR10(root='data', train=False,
		download=True, transform=transform)
	return testset

def make_digit_dataset(flatten=True):
	data, target = load_digits(return_X_y=True)
	data = data.reshape(len(data), -1) / 255.
	return data, target

def create_torch_model(device):
	model = resnet.resnet20()
	weights = torch.load(TORCH_WEIGHTS, map_location=device)
	model = nn.DataParallel(model)
	model.load_state_dict(weights['state_dict'])
	model = model.module
	model.eval()
	model.to(device)
	print('Created PyTorch model:\t', model._get_name())
	print(' * Dataset:\t\t CIFAR10')
	print(' * Best Test prec:\t', weights['best_prec1'])
	return model

def create_tf_model_intermediate():
    dense1 = tf.keras.layers.Dense(64, activation='elu')
    dense2 = tf.keras.layers.Dense(64, activation='elu')
    dense3 = tf.keras.layers.Dense(10, activation='softmax')
    # in order to access the intermediate embedding, split the model into two models
    # model_embd will embed samples
    model_embd = tf.keras.Sequential([dense1])
    # model_head can make predictions based on the embedding from model_embd
    model_head = tf.keras.Sequential([dense2, dense3])
    whole_model = tf.keras.Sequential([model_embd, model_head])
    return model_embd, model_head, whole_model

def create_decision_tree(train_x, train_y, max_depth=8):
	d_tree = DecisionTreeClassifier(max_depth=max_depth)
	d_tree = d_tree.fit(train_x, train_y)
	test_score = d_tree.score(train_x, train_y)
	print('Created decision tree')
	print(' * Depth:\t\t', d_tree.get_depth())
	print(' * Dataset:\t\t MNIST')
	print(' * Train score:\t\t', test_score)
	return d_tree

def create_random_forest(train_x, train_y, n_estimators=100):
	r_forest = RandomForestClassifier(n_estimators)
	r_forest = r_forest.fit(train_x, train_y)
	test_score = r_forest.score(train_x, train_y)
	print('Created random forest')
	print(' * No. of Estimators:\t', n_estimators)
	print(' * Dataset:\t\t MNIST')
	print(' * Train score:\t\t', test_score)
	return r_forest

def create_kn_neighbors(train_x, train_y, k=10):
	k_neighbors = KNeighborsClassifier(k)
	k_neighbors = k_neighbors.fit(train_x, train_y)
	test_score = k_neighbors.score(train_x, train_y)
	print('Created knn classifier')
	print(' * No. of Neighbors:\t', k)
	print(' * Dataset:\t\t MNIST')
	print(' * Train score:\t\t', test_score)
	return k_neighbors


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 50, 5, 1)
        self.conv2 = nn.Conv2d(50, 100, 5, 1)
        self.fc1 = nn.Linear(4*4*100, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.contiguous().view(-1, 4*4*100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict_numpy(self, x):
    	# Can be used as the prediction wrapper for DeepView
    	with torch.no_grad():
    		is_cuda = next(self.parameters()).is_cuda
    		device = 'cuda:0' if is_cuda else 'cpu'
    		x = np.array(x, dtype=np.float32)
    		x = torch.from_numpy(x).to(device)
    		prob = self.forward(x).exp()
    		prediction = prob.cpu().numpy()
    	return prediction


def add_backdoor(image):
    if len(image.shape) == 2:
        image[0,-1] = 1
        image[1,-2] = 1
        image[0,-2] = 0
        image[1,-1] = 0
    elif len(image.shape) == 3:
        image[0,0,-1] = 1
        image[0,1,-2] = 1
        image[0,0,-2] = 0
        image[0,1,-1] = 0

def train_backdoor(model, device, trainloader, optimizer, epoch, log_interval=10, backd_a=8, backd_t=1, n_backd=600):
    model.train()
    batch_size = trainloader.batch_size
    n_data = len(trainloader.dataset) 

    # first, get the indices of the first n_backd items of class backd_a
    idx_bc  = np.zeros(n_backd, dtype=int)
    
    n = 0
    for i in range(n_data):
        if trainloader.dataset.__getitem__(i)[1] == backd_a:
            idx_bc[n] = i
            n += 1
            if n == n_backd:
                break

    
    for batch_idx, (data, target) in enumerate(trainloader):
        # add backdoor for the required samples
        curr_idx = range(batch_size*batch_idx, batch_size*batch_idx + batch_size)
        for i in curr_idx:
            if (idx_bc == i).any():
                add_backdoor(data[i-batch_size*batch_idx])
                target[i-batch_size*batch_idx] = backd_t

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.contiguous().view_as(pred)).sum().item()
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def mnist_visualization(image, point2d, pred, label=None):
    '''
    Demo visualization method for visualizing a 64-dim vector as an 8x8-image.
    Used in demo for MNIST-Datapoints.
    '''
    f, a = plt.subplots()
    a.set_title('Prediction: %d' % pred)
    a.imshow(image.reshape(8,8))