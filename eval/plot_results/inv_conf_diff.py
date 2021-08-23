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


def main(args):
    dataset = args.dataset
    start = args.s
    end = args.e
    p = args.p

    #%%

    data = np.array([])
    # load data from evaluation.json
    content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
    for epoch in range(start, end, p):
        eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation.json")
        with open(eval_path, "r") as f:
            eval = json.load(f)
        nn_train = round(eval["inv_conf_train"], 3)
        nn_test = round(eval["inv_conf_test"], 3)

        if len(data)==0:
            data = np.array([[dataset, "DVI", "Train", "{}".format(str(epoch//p)), nn_train]])
        else:
            data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
        data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

    #%%
    # load data from evaluation_step2.json
    content_path = "E:\\DVI_exp_data\\TemporalExp\\resnet18_{}".format(dataset)
    for epoch in [1, 2, 3, 4, 7]:
        eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2.json")
        with open(eval_path, "r") as f:
            eval = json.load(f)
        nn_train = round(eval["inv_conf_train"], 3)
        nn_test = round(eval["inv_conf_test"], 3)
        if epoch>5:
            i=5
        else:
            i=epoch
        data = np.concatenate((data, np.array([[dataset, "DVI-temporal", "Train", "{}".format(str(i)), nn_train]])), axis=0)
        data = np.concatenate((data, np.array([[dataset, "DVI-temporal", "Test", "{}".format(str(i)), nn_test]])), axis=0)

    content_path = "E:\\xianglin\\git_space\\umap_exp\\results"
    # pca
    curr_path = os.path.join(content_path, "pca")
    for epoch in range(start, end, p):
        eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
        with open(eval_path, "r") as f:
            eval = json.load(f)
        nn_train = round(eval[7], 3)
        nn_test = round(eval[9], 3)

        data = np.concatenate((data, np.array([[dataset, "PCA", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
        data = np.concatenate((data, np.array([[dataset, "PCA", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)
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
        eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
        with open(eval_path, "r") as f:
            eval = json.load(f)
        nn_train = round(eval[7], 3)
        nn_test = round(eval[9], 3)

        data = np.concatenate((data, np.array([[dataset, "UMAP", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
        data = np.concatenate((data, np.array([[dataset, "UMAP", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

    #%%

    col = np.array(["dataset", "method", "type", "period", "eval"])
    df = pd.DataFrame(data, columns=col)
    df[["period"]] = df[["period"]].astype(int)
    df[["eval"]] = df[["eval"]].astype(float)

    #%%

    pal20c = sns.color_palette('tab20c', 20)
    sns.palplot(pal20c)
    hue_dict = {
        "DVI": pal20c[4],
        "DVI-temporal": pal20c[6],
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
    hue_list = ["DVI", "DVI-temporal", "UMAP", "PCA"]

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
    axs[0].set_ylim(0., max_*1.1)
    axs[0].set_title("Train")
    axs[1].set_title("Test")

    (fg.despine(bottom=True)
     .set_xticklabels(['Begin', 'Early', 'Mid', 'Late', 'End'])
     .set_axis_labels("", "Inverse Confidence Difference")
     )
    fg.fig.suptitle(dataset)

    #%%

    fg.savefig(
        "inv_conf_{}.png".format(dataset),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw inverse confidence difference plot for different datasets.")

    parser.add_argument("--dataset", type=str)
    parser.add_argument("-s", type=int)
    parser.add_argument('-e', type=int)
    parser.add_argument('-p', type=int)
    args = parser.parse_args()
    main(args)

