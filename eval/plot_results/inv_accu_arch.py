#%% md

# plot mnist train nearest neighbor preserving property

#%%

# import modules
import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns


arch = "resnet50"
start = 40
end = 240
p = 40

#%%

data = np.array([])
# load data from evaluation.json
content_path = "E:\\DVI_exp_data\\resnet50_cifar10"
for epoch in range(start, end, p):
    eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation.json")
    with open(eval_path, "r") as f:
        eval = json.load(f)
    nn_train = round(eval["inv_acc_train"], 3)
    nn_test = round(eval["inv_acc_test"], 3)

    if len(data)==0:
        data = np.array([[arch, "DVI", "Train", "{}".format(str(epoch//p)), nn_train]])
    else:
        data = np.concatenate((data, np.array([[arch, "DVI", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
    data = np.concatenate((data, np.array([[arch, "DVI", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

#%%

content_path = "E:\\xianglin\\git_space\\umap_exp\\results"
# pca
curr_path = os.path.join(content_path, "pca")
for epoch in range(start, end, p):
    eval_path = os.path.join(curr_path, "{}_{}".format(arch, epoch), "exp_result.json")
    with open(eval_path, "r") as f:
        eval = json.load(f)
    nn_train = round(eval[6], 3)
    nn_test = round(eval[8], 3)

    data = np.concatenate((data, np.array([[arch, "PCA", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
    data = np.concatenate((data, np.array([[arch, "PCA", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)
# tsne
# no applicable
# curr_path = os.path.join(content_path, "tsne")
# for epoch in range(start, end, p):
#     eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
#     with open(eval_path, "r") as f:
#         eval = json.load(f)
#     nn_train = round(eval[], 3)
#
#     data = np.concatenate((data, np.array([[dataset, "TSNE", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)

# umap
curr_path = os.path.join(content_path, "umap")
for epoch in range(start, end, p):
    eval_path = os.path.join(curr_path, "{}_{}".format(arch, epoch), "exp_result.json")
    with open(eval_path, "r") as f:
        eval = json.load(f)
    nn_train = round(eval[6], 3)
    nn_test = round(eval[8], 3)

    data = np.concatenate((data, np.array([[arch, "UMAP", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
    data = np.concatenate((data, np.array([[arch, "UMAP", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

#%%

col = np.array(["arch", "method", "type", "period", "eval"])
df = pd.DataFrame(data, columns=col)
df[["period"]] = df[["period"]].astype(int)
df[["eval"]] = df[["eval"]].astype(float)

#%%

pal20c = sns.color_palette('tab20c', 20)
sns.palplot(pal20c)
hue_dict = {
    "DVI": pal20c[4],
    "UMAP": pal20c[0],
    # "TSNE": pal20c[8],
    "PCA": pal20c[12],

}
sns.palplot([hue_dict[i] for i in hue_dict.keys()])

#%%

axes = {'labelsize': 14,
        'titlesize': 14,}
mpl.rc('axes', **axes)
mpl.rcParams['xtick.labelsize'] = 14

# hue_list = ["TSNE", "parametric-tsne", "umap-learn",  'direct', "network", "autoencoder", 'vae', 'ae_only', "PCA"]
# hue_list = ["DVI", "UMAP", "TSNE", "PCA"]
hue_list = ["DVI", "UMAP", "PCA"]

#%%

fg = sns.catplot(
    x="period",
    y="eval",
    hue="method",
    hue_order=hue_list,
    # order = [1, 2, 3, 4, 5],
    # row="method",
    col="type",
    ci=0.001,
    height=3, #2.65,
    aspect=2.5,#3,
    data=df,
    kind="bar",
    palette=[hue_dict[i] for i in hue_list],
    legend=True
)

axs = fg.axes[0]
max_ = df["eval"].max()
# min_ = df["eval"].min()
axs[0].set_ylim(0, max_*1.1)
axs[0].set_title("Train")
axs[1].set_title("Test")

(fg.despine(bottom=True)
 .set_xticklabels(['Begin', 'Early', 'Mid', 'Late', 'End'])
 .set_axis_labels("", "Inverse Accuracy")
 )
fg.fig.suptitle("cifar10_2048")

#%%

fg.savefig(
    "inv_accu_{}.png".format(2048),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.0,
    transparent=True,
)


