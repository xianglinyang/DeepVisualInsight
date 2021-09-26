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
            nn_train = round(eval["inv_acc_train"], 3)
            nn_test = round(eval["inv_acc_test"], 3)

            if len(data)==0:
                data = np.array([[dataset, "DVI", "Train", "DVI-T-Train", "{}".format(str(epoch//p)), nn_train]])
            else:
                data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-T-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-T-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

        #%%
        #%%
        # load data from evaluation_step2.json
        content_path = "E:\\DVI_exp_data\\TemporalExp\\resnet18_{}".format(dataset)
        for epoch in [1, 2, 3, 4, 7]:
            eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval["inv_acc_train"], 3)
            nn_test = round(eval["inv_acc_test"], 3)
            if epoch>5:
                i=5
            else:
                i=epoch
            data = np.concatenate((data, np.array([[dataset, "DVI-temporal", "Train", "DVI-Train", "{}".format(str(i)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "DVI-temporal", "Test", "DVI-Test", "{}".format(str(i)), nn_test]])), axis=0)

        # parametric umap
        content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
        for epoch in range(start, end, p):
            eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval["inv_acc_train"], 3)
            nn_test = round(eval["inv_acc_test"], 3)

            data = np.concatenate((data, np.array([[dataset, "parametricUmap", "Train", "parametricUmap-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "parametricUmap", "Test", "parametricUmap-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

        content_path = "E:\\xianglin\\git_space\\umap_exp\\results"
        # pca
        curr_path = os.path.join(content_path, "pca")
        for epoch in range(start, end, p):
            eval_path = os.path.join(curr_path, "{}_{}".format(dataset, epoch), "exp_result.json")
            with open(eval_path, "r") as f:
                eval = json.load(f)
            nn_train = round(eval[6], 3)
            nn_test = round(eval[8], 3)

            data = np.concatenate((data, np.array([[dataset, "PCA", "Train", "PCA-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "PCA", "Test", "PCA-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)
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
            nn_train = round(eval[6], 3)
            nn_test = round(eval[8], 3)

            data = np.concatenate((data, np.array([[dataset, "UMAP", "Train", "UMAP-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "UMAP", "Test", "UMAP-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

        #%%
        df_tmp = pd.DataFrame(data, columns=col)
        df = df.append(df_tmp, ignore_index=True)
        df[["period"]] = df[["period"]].astype(int)
        df[["eval"]] = df[["eval"]].astype(float)

    #%%
    df.to_excel("PPR.xlsx")

    pal20c = sns.color_palette('tab20c', 20)
    sns.set_theme(style="whitegrid", palette=pal20c)
    # sns.palplot(pal20c)
    hue_dict = {
        "DVI-Train": pal20c[0],
        "parametricUmap-Train": pal20c[12],
        "UMAP-Train": pal20c[4],
        "PCA-Train": pal20c[8],
        "DVI-Test": pal20c[3],
        "parametricUmap-Test": pal20c[15],
        "UMAP-Test": pal20c[7],
        "PCA-Test": pal20c[11],
    }
    sns.palplot([hue_dict[i] for i in hue_dict.keys()])
    #%%

    axes = {'labelsize': 9,
            'titlesize': 9,}
    mpl.rc('axes', **axes)
    mpl.rcParams['xtick.labelsize'] = 9

    # hue_list = ["DVI-Train", "DVI-Test", "DVI-T-Train", "DVI-T-Test", "UMAP-Train", "UMAP-Test", "PCA-Train", "PCA-Test"]
    hue_list = ["DVI-Train", "DVI-Test", "parametricUmap-Train", "parametricUmap-Test", "UMAP-Train", "UMAP-Test", "PCA-Train", "PCA-Test"]
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
    sns.move_legend(fg, "lower center", bbox_to_anchor=(.42, 0.92), ncol=4, title=None, frameon=False)
    mpl.pyplot.setp(fg._legend.get_texts(), fontsize='9')

    axs = fg.axes[0]
    max_ = df["eval"].max()
    # min_ = df["eval"].min()
    axs[0].set_ylim(0., max_*1.1)
    axs[0].set_title("MNIST")
    axs[1].set_title("FMNIST")
    axs[2].set_title("CIFAR-10")
    # # iterate through axes
    # for ax in axs.ravel():
    #     # add annotations
    #     for c in ax.containers:
    #         labels = [f'{(v.get_height()):.2f}' for v in c]
    #         ax.bar_label(c, labels=labels, label_type='edge')
    #     ax.margins(y=0.2)

    (fg.despine(bottom=False, right=False, left=False, top=False)
     .set_xticklabels(['Begin', 'Early', 'Mid', 'Late', 'End'])
     .set_axis_labels("", "PPR")
     )
    # fg.fig.suptitle("NN preserving property")

    #%%
    fg.savefig(
        "inv_accu.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Draw inverse accuracy plot for different datasets.")
    #
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("-s", type=int)
    # parser.add_argument('-e', type=int)
    # parser.add_argument('-p', type=int)
    # args = parser.parse_args()
    main()

