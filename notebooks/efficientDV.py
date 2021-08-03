
import umap
import sys
import time
import numba
import numpy as np

from deepvisualinsight.MMS import MMS
# prepare training data
content_path = "E:\\DVI_exp_data\\resnet18_cifar10"
sys.path.append(content_path)
from Model.model import *
net = resnet18()

classes = ("airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
mms = MMS(content_path, net, 10, 200, 10, 512, 10, classes, cmap="tab10", resolution=100, neurons=256, verbose=1, temporal=False, split=-1, advance_border_gen=True, alpha=0.8, attack_device="cuda:0")

# shape (50000,3,32,32)
training_data = mms.training_data.cpu().numpy()[:500]
# shape (50000)
training_labels = mms.training_labels.cpu().numpy()[:500]
training_data = training_data.reshape(500,-1)



# # @numba.njit()
# def euclidean_numba(x1,x2):
#     return np.linalg.norm(x1-x2)
# # umap with euclidean distance, with numba.njit and nndescent
# reducer = umap.UMAP(n_components=2, metric=euclidean_numba)
# t0 = time.time()
# out = reducer.fit_transform(training_data)
# t1 = time.time()
# print("{:.2f} seconds.".format(t1-t0))
import socket               # 导入 socket 模块
@numba.njit()
def euclidean_server(x1,x2):
    s = socket.socket()         # 创建 socket 对象
    host = socket.gethostname() # 获取本地主机名
    port = 12345                # 设置端口号

    s.connect((host, port))
    query = np.vstack((x1,x2))
    s.send(query.tostring())
    ans = s.recv(64)
    s.close()
    ans = np.fromstring(ans, dtype=np.float64)[0]
    return ans
# a = euclidean_server(training_data[0],training_data[1])

reducer = umap.UMAP(n_components=2, metric=euclidean_server)
t0 = time.time()
out = reducer.fit_transform(training_data)
t1 = time.time()
print("{:.2f} seconds.".format(t1-t0))

