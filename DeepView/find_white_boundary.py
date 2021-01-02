# using deepview
import os
import numpy as np
from deepview import DeepView
# load modules
import matplotlib.pyplot as plt

# load modules
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import numpy as np
import os
import math

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 36
data_augmentation = True
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 9

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
# we only consider v1 here
version = 1

# Computed depth from supplied model parameter n
depth = n * 6 + 2
# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)
print("model_type:", model_type)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# remember to reset embedder
encoder_location = "parametric_umap_models\\ResNet56v1\\"

data_dir = os.path.join(os.getcwd(), 'high_output')
n_epoches = np.arange(100, 210, 10)

for n_epoch in n_epoches:
    datapath = os.path.join(data_dir, "train_{:03d}.npy".format(n_epoch))
    train_data = np.load(datapath)

    # load encoder
    encoder_path = os.path.join(encoder_location, "encoder_{:03d}".format(n_epoch))
    encoder = tf.keras.models.load_model(encoder_path)
    print("Keras encoder model loaded from {}".format(encoder_path))
    # load decoder
    decoder_path = os.path.join(encoder_location, "decoder_{:03d}".format(n_epoch))
    decoder = tf.keras.models.load_model(decoder_path)
    print("Keras decoder model loaded from {}".format(decoder_path))


    # ---------------------load dataset------------------------------

    def pred_wrapper(x):
        model_name = 'cifar10_ResNet56v1_model.{:03d}.h5'.format(n_epoch)
        modelpath = os.path.join(os.path.join(os.getcwd(), 'resnet_models'), model_name)
        load_model = tf.keras.models.load_model(modelpath)

        # load fully connect layer
        fc_input = tf.keras.layers.Input(shape=(64,))
        logits = fc_input
        logits = load_model.get_layer("dense")(logits)
        fc_model = Model(inputs=fc_input, outputs=logits)

        output = fc_model(x).cpu().numpy()
        # probabilities = tf.nn.softmax(output, axis=-1).cpu().numpy()
        return output


    def visualization(image, point2d, pred, label=None, title=None):
        f, a = plt.subplots()
        a.set_title(title)
        a.imshow(image.transpose([1, 2, 0]))


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # ---------------------deepview------------------------------
    batch_size = 200
    max_samples = 50000
    data_shape = (64,)
    n = 5
    lam = 0
    resolution = 100
    cmap = 'tab10'
    title = 'ResNet-56 - CIFAR10 GAP layer-parametric umap autoencoder'

    deepview = DeepView(pred_wrapper, classes, max_samples, batch_size,
                        data_shape, n, lam, resolution, cmap, title=title, data_viz=visualization,
                        clip_certainty=0.25, metric="parametricUmap", encoder=encoder, decoder=decoder)

    deepview.add_samples(train_data[:100], np.argmax(y_train, axis=1)[:100])
    deepview.savefig("result\\evaluation\\parametricUmap\\deepview_train_{:03d}.png".format(n_epoch))

data_dir = os.path.join(os.getcwd(), 'high_output')
encoder_location = "parametric_umap_models\\ResNet56v1\\"

# n_epoches = np.arange(2, 38, 2)
# for n_epoch in n_epoches:
#     datapath = os.path.join(data_dir, "train_{:03d}.npy".format(n_epoch))
#     train_data = np.load(datapath)
#
#     # load encoder
#     encoder_path = os.path.join(encoder_location, "encoder_{:03d}".format(n_epoch))
#     encoder = tf.keras.models.load_model(encoder_path)
#     print("Keras encoder model loaded from {}".format(encoder_path))
#
#     z = encoder(train_data)
#     #     ??? .cpu.numpy()
#     print(type(z))
#
#     fig, ax = plt.subplots(ncols=1, figsize=(10, 8))
#     sc = ax.scatter(
#         z[:, 0],
#         z[:, 1],
#         c=np.argmax(y_train, axis=1),
#         cmap="tab10",
#         s=0.1,
#         alpha=0.5,
#         rasterized=True,
#     )
#     ax.axis('equal')
#     ax.set_title("parametric UMAP autoencoder embeddings-training data", fontsize=20)
#     fig.savefig("result/evaluation/parametricUmap/train_{:03d}".format(n_epoch))