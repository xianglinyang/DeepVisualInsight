# import modules
from deepvisualinsight.MMS import MMS
from deepvisualinsight import utils
from deepvisualinsight.backend import get_alpha
import sys
import os
import numpy as np
import time
import torch
import json
import tensorflow as tf
import time
from scipy.special import softmax


content_path = "E:\\DVI_exp_data\\RQ1\\fix_border"
sys.path.append(content_path)

from Model.model import *
net = resnet18()
classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

mms = MMS(content_path, net, 40, 40, 60, 512, 10, classes, cmap="tab10", resolution=100, neurons=256,
          verbose=1, temporal=False, split=-1, advance_border_gen=True, alpha=0.6, attack_device="cuda:0")

# hyperparameters
EPOCH = 40
# mms.data_preprocessing()
# #%%
model_location = os.path.join(mms.model_path, "Epoch_{:d}".format(EPOCH), "subject_model.pth")
net.load_state_dict(torch.load(model_location, map_location=torch.device("cpu")))
net = net.to(mms.device)
net.eval()

repr_model = torch.nn.Sequential(*(list(net.children())[-1:]))
repr_model = repr_model.to(mms.device)
repr_model = repr_model.eval()
#%%
for param in repr_model.parameters():
    param.requires_grad = False
#%%
train_data = mms.get_epoch_train_repr_data(EPOCH)
border_points = mms.get_epoch_border_centers(EPOCH)
fitting_data = np.concatenate((train_data, border_points), axis=0)
#%%
alpha = get_alpha(repr_model, fitting_data, temperature=.01, device=torch.device("cuda:0"), verbose=1)
#%%

mms.prepare_visualization_for_all()
mms.save_evaluation("beta=1.")