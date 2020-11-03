from deepview import DeepView
import numpy as np
import time

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier


# load and process data
data, target = load_digits(return_X_y=True)
data = data.reshape(len(data), -1) / 255.
classes = np.arange(10)

# create a random forest classifier
n_trees = 100
model = RandomForestClassifier(n_trees)
model = model.fit(data, target)
test_score = model.score(data, target)
print('Created random forest')
print(' * No. of Estimators:\t', n_trees)
print(' * Train score:\t\t', test_score)

# create a wrapper function for deepview
# Here, because random forest can handle numpy lists and doesn't
# need further preprocessing or conversion into a tensor datatype
pred_wrapper = DeepView.create_simple_wrapper(model.predict_proba)

# this is the alternative way of defining the prediction wrapper
# for deep learning frameworks. In this case it's PyTorch.
def torch_wrapper(x):
    with torch.no_grad():
        x = np.array(x, dtype=np.float32)
        tensor = torch.from_numpy(x).to(device)
        pred = model(tensor).cpu().numpy()
    return pred

# --- Deep View Parameters ----
batch_size = 32
max_samples = 500
data_shape = (64,)
resolution = 100
N = 10
lam = 0.64
cmap = 'tab10'
# to make shure deepview.show is blocking,
# disable interactive mode
interactive = False
title = 'Forest - MNIST'

deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape, 
	N, lam, resolution, cmap, interactive, title)

deepview.add_samples(data[:50], target[:50])
deepview.show()