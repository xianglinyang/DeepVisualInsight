import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np

class TorchModel(nn.Module):

	def __init__(self):
		super(TorchModel, self).__init__()
		self.conv1  = nn.Conv2d(3, 128, 3, stride=2)
		self.pool   = nn.MaxPool2d(2, 2)
		self.conv2  = nn.Conv2d(128, 256, 3, stride=2)
		self.conv3  = nn.Conv2d(256, 512, 3, stride=2)
		self.dense1 = nn.Linear(1*1*512, 256)
		self.dense2 = nn.Linear(256, 10)

		self.objective = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.parameters())

	def forward(self, x):
		# 15 x 15 x 128
		x = F.relu(self.conv1(x))
		# 7 x 7 x 128
		x = self.pool(x)
		# 3 x 3 x 256
		x = F.relu(self.conv2(x))
		# 1 x 1 x 512
		x = F.relu(self.conv3(x))
		x = x.view(-1, 1*1*512)
		x = F.relu(self.dense1(x))
		x = self.dense2(x)
		return x

	def train(self, dataset, device, epochs=2):
		for epoch in range(epochs):

			epoch_loss = []

			for i, (x, y) in enumerate(dataset):
				# set gradients to zero
				self.optimizer.zero_grad()
				# forward pass
				outputs = self(x.to(device))
				loss = self.objective(outputs, y.to(device))
				# backward pass
				loss.backward()
				self.optimizer.step()

				epoch_loss.append(loss.item())

				#if i % 1000 == 0:
				#	print('Batch, %d finished with loss: %.3f' % (i+1, loss))
			print('Epoch finished, mean loss: %.3f' % np.mean(epoch_loss))


	def save(self, path):
		torch.save(self.state_dict(), path)


	def load(self, path):
		weights = torch.load(path)
		self.load_state_dict(weights)

		