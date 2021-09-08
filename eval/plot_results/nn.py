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


def main():
    # dataset = args.dataset
    # start = args.s
    # end = args.e
    # p = args.p
    datasets = ["mnist", "fmnist", "cifar10"]
    starts = [4, 10, 40]
    ends = [24, 60, 240]
    periods = [4, 10, 40]
    col = np.array(["dataset", "method", "type", "hue", "period", "eval"])
    df = pd.DataFrame({}, columns=col)

    for i in range(3):
        dataset = datasets[i]
        start = starts[i]
        end = ends[i]
        p = periods[i]

        data = np.array([])
        # load data from evaluation.json
        content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
        for epoch in range(start, end, p):
            eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval["nn_train_15"], 3)
            nn_test = round(eval["nn_test_15"], 3)

            if len(data)==0:
                data = np.array([[dataset, "DVI", "Train","DVI-Train", "{}".format(str(epoch//p)), nn_train]])
            else:
                data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

        # load data from evaluation.json
        content_path = "E:\\DVI_exp_data\\TemporalExp\\resnet18_{}".format(dataset)
        for epoch in [1, 2, 3, 4, 7]:
            eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval["nn_train_15"], 3)
            nn_test = round(eval["nn_test_15"], 3)
            if epoch>5:
                i=5
            else:
                i=epoch
            data = np.concatenate((data, np.array([[dataset, "DVI-temporal", "Train", "DVI-temporal-Train", "{}".format(str(i)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "DVI-temporal", "Test", "DVI-temporal-Test", "{}".format(str(i)), nn_test]])), axis=0)

        #%%

        content_path = "E:\\xianglin\\git_space\\umap_exp\\results"
        # pca
        curr_path = os.path.join(content_path, "pca")
        for epoch in range(start, end, p):
            eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval[1], 3)
            nn_test = round(eval[4], 3)

            data = np.concatenate((data, np.array([[dataset, "PCA", "Train", "PCA-Train",  "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "PCA", "Test", "PCA-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)
        # tsne
        curr_path = os.path.join(content_path, "tsne")
        for epoch in range(start, end, p):
            eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval[1], 3)

            data = np.concatenate((data, np.array([[dataset, "TSNE", "Train", "TSNE-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)

        # umap
        curr_path = os.path.join(content_path, "umap")
        for epoch in range(start, end, p):
            eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval[1], 3)
            nn_test = round(eval[4], 3)

            data = np.concatenate((data, np.array([[dataset, "UMAP", "Train", "UMAP-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "UMAP", "Test", "UMAP-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

        #%%
        df_tmp = pd.DataFrame(data, columns=col)
        # df_tmp[["period"]] = df_tmp[["period"]].astype(int)
        # df_tmp[["eval"]] = df_tmp[["eval"]].astype(float)
        df = df.append(df_tmp, ignore_index=True)
        df[["period"]] = df[["period"]].astype(int)
        df[["eval"]] = df[["eval"]].astype(float)

    #%%

    pal20c = sns.color_palette('tab20c', 20)
    sns.palplot(pal20c)
    hue_dict = {
        "DVI-Train": pal20c[0],
        "DVI-temporal-Train": pal20c[4],
        "UMAP-Train": pal20c[8],
        "TSNE-Train": pal20c[16],
        "PCA-Train": pal20c[12],
        "DVI-Test": pal20c[3],
        "DVI-temporal-Test": pal20c[7],
        "UMAP-Test": pal20c[11],
        # "TSNE": pal20c[8],
        "PCA-Test": pal20c[14],

    }
    sns.palplot([hue_dict[i] for i in hue_dict.keys()])

    #%%

    axes = {'labelsize': 9,
            'titlesize': 9,}
    mpl.rc('axes', **axes)
    mpl.rcParams['xtick.labelsize'] = 9

    hue_list = ["DVI-Train", "DVI-Test", "DVI-temporal-Train", "DVI-temporal-Test", "UMAP-Train", "UMAP-Test", "PCA-Train", "PCA-Test", "TSNE-Train"]

    #%%
    # sns.set_style("dark")
    # sns.set_style('darkgrid')
    # sns.set(style='ticks')

    fg = sns.catplot(
        x="period",
        y="eval",
        hue="hue",
        hue_order=hue_list,
        # order = [1, 2, 3, 4, 5],
        # row="method",
        col="dataset",
        ci=0.001,
        height=2.5, #2.65,
        aspect=1.0,#3,
        data=df,
        kind="bar",
        palette=[hue_dict[i] for i in hue_list],
        legend=True
    )
    sns.move_legend(fg, "lower center", bbox_to_anchor=(.42, 0.92), ncol=5, title=None, frameon=False)
    mpl.pyplot.setp(fg._legend.get_texts(), fontsize='9')

    axs = fg.axes[0]
    max_ = df["eval"].max()
    # min_ = df["eval"].min()
    axs[0].set_ylim(0., max_*1.1)
    axs[0].set_title("MNIST")
    axs[1].set_title("FMNIST")
    axs[2].set_title("CIFAR-10")

    (fg.despine(bottom=False, right=False, left=False, top=False)
     .set_xticklabels(['Begin', 'Early', 'Mid', 'Late', 'End'])
     .set_axis_labels("Period", "NN Preserving")
     )
    # fg.fig.suptitle("NN preserving property")

    #%%
    fg.savefig(
        "nn.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Draw nearest neighbor plot for different datasets.")
    #
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("-s", type=int)
    # parser.add_argument('-e', type=int)
    # parser.add_argument('-p', type=int)
    # args = parser.parse_args()
    main()

