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
    name_dict = {"cifar10": "resnet18", "fmnist": "FASHIONMNIST", "mnist": "MNIST", "resnet50": "CIFAR10"}

    data = np.array([])
    # load data from evaluation.json
    for epoch in range(start, end, p):
        nn_train = .0
        nn_test = .0
        for i in range(1, 11, 1):
            content_path = "E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}".format(i)
            eval_path = os.path.join(content_path,"{}".format(dataset), "Model", "Epoch_{}".format(epoch), "evaluation.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train += round(eval["inv_conf_train"], 4)
            nn_test += round(eval["inv_conf_test"], 4)
        nn_train = round(nn_train / 10, 3)
        nn_test = round(nn_test / 10, 3)
        if len(data)==0:
            data = np.array([[dataset, "DVI", "Train", "{}".format(str(epoch//p)), nn_train]])
        else:
            data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
        data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

    content_path = "E:\\xianglin\\git_space\\DeepView\DVI_exp\\batch_run_results"

    for epoch in range(start, end, p):
        nn_train = .0
        # nn_test = .0
        for i in range(1, 11, 1):
            curr_path = os.path.join(content_path, "{}".format(i))
            eval_path = os.path.join(curr_path, "{}_{}".format(name_dict[dataset], epoch), "exp_result.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train += round(eval[7], 4)
        # nn_test = round(eval[4], 3)
        nn_train = round(nn_train / 10, 3)
        data = np.concatenate((data, np.array([[dataset, "DeepView", "Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
        # data = np.concatenate((data, np.array([[dataset, "DeepView", "Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)


    col = np.array(["dataset", "method", "type", "period", "eval"])
    df = pd.DataFrame(data, columns=col)
    df[["period"]] = df[["period"]].astype(int)
    df[["eval"]] = df[["eval"]].astype(float)

    #%%

    pal20c = sns.color_palette('tab20c', 20)
    sns.palplot(pal20c)
    hue_dict = {
        "DVI": pal20c[4],
        # "UMAP": pal20c[0],
        # "TSNE": pal20c[8],
        "DeepView": pal20c[12],

    }
    sns.palplot([hue_dict[i] for i in hue_dict.keys()])

    axes = {'labelsize': 14,
            'titlesize': 14,}
    mpl.rc('axes', **axes)
    mpl.rcParams['xtick.labelsize'] = 14

    # hue_list = ["TSNE", "parametric-tsne", "umap-learn",  'direct', "network", "autoencoder", 'vae', 'ae_only', "PCA"]
    hue_list = ["DVI", "DeepView"]

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
     .set_axis_labels("", "Inverse Accuracy property")
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
    parser = argparse.ArgumentParser(description="Draw nearest neighbor plot for different datasets.")

    parser.add_argument("--dataset", type=str)
    parser.add_argument("-s", type=int)
    parser.add_argument('-e', type=int)
    parser.add_argument('-p', type=int)
    args = parser.parse_args()
    main(args)

