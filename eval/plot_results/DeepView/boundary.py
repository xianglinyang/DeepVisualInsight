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
    name_dict = {"cifar10": "resnet18", "fmnist": "FASHIONMNIST", "mnist": "MNIST", "resnet50": "CIFAR10"}
    datasets = ["mnist", "fmnist", "cifar10"]
    starts = [4, 10, 40]
    ends = [24, 60, 240]
    periods = [4, 10, 40]
    k_neighbors = [10, 15, 20]
    col = np.array(["dataset", "method", "type", "hue", "k", "period", "eval"])
    df = pd.DataFrame({}, columns=col)

    for k in k_neighbors:
        for i in range(3):
            dataset = datasets[i]
            start = starts[i]
            end = ends[i]
            p = periods[i]

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
                    nn_train += round(eval["bound_train_{}".format(k)], 4)
                    nn_test += round(eval["bound_test_{}".format(k)], 4)
                nn_train = round(nn_train / 10, 3)
                nn_test = round(nn_test / 10, 3)
                if len(data)==0:
                    data = np.array([[dataset, "DVI-T", "Train", "DVI-T-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])
                else:
                    data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-T-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-T-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)

            # load data from evaluation_step2.json
            for epoch in range(start, end, p):
                nn_train = .0
                nn_test = .0
                for i in range(1, 11, 1):
                    content_path = "E:\\DVI_exp_data\\DeepViewExp\\multi_run\\{}".format(i)
                    eval_path = os.path.join(content_path,"{}".format(dataset), "Model", "Epoch_{}".format(epoch), "evaluation_step2.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train += round(eval["bound_train_{}".format(k)], 4)
                    nn_test += round(eval["bound_test_{}".format(k)], 4)
                nn_train = round(nn_train / 10, 3)
                nn_test = round(nn_test / 10, 3)
                data = np.concatenate((data, np.array([[dataset, "DVI", "Train",  "DVI-Train", "{}".format(k),"{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)

            content_path = "E:\\xianglin\\git_space\\DeepView\DVI_exp\\batch_run_results"

            for epoch in range(start, end, p):
                nn_train = .0
                nn_test = .0
                for i in range(1, 11, 1):
                    curr_path = os.path.join(content_path, "{}".format(i))
                    eval_path = os.path.join(curr_path, "{}_{}".format(name_dict[dataset], epoch), "exp_result.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train += round(eval[k//5+1], 4)
                    nn_test += round(eval[k//5+9], 4)
                nn_train = round(nn_train / 10, 3)
                nn_test = round(nn_test / 10, 3)
                data = np.concatenate((data, np.array([[dataset, "DeepView", "Train", "DeepView-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "DeepView", "Test", "DeepView-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)


            df_tmp = pd.DataFrame(data, columns=col)
            df = df.append(df_tmp, ignore_index=True)
            df[["period"]] = df[["period"]].astype(int)
            df[["k"]] = df[["k"]].astype(int)
            df[["eval"]] = df[["eval"]].astype(float)

    #%%
    df.to_excel("boundary.xlsx")

    pal20c = sns.color_palette('tab20c', 20)
    sns.palplot(pal20c)
    hue_dict = {
        "DVI-Train": pal20c[0],
        "DVI-T-Train": pal20c[4],
        "DeepView-Train": pal20c[8],
        "DVI-Test": pal20c[1],
        "DVI-T-Test": pal20c[5],
        "DeepView-Test": pal20c[9],

    }
    sns.palplot([hue_dict[i] for i in hue_dict.keys()])

    axes = {'labelsize': 14,
            'titlesize': 14,}
    mpl.rc('axes', **axes)
    mpl.rcParams['xtick.labelsize'] = 14

    # hue_list = ["TSNE", "parametric-tsne", "umap-learn",  'direct', "network", "autoencoder", 'vae', 'ae_only', "PCA"]
    hue_list = ["DVI-Train", "DVI-Test", "DVI-T-Train", "DVI-T-Test", "DeepView-Train", "DeepView-Test"]

    #%%

    fg = sns.catplot(
        x="period",
        y="eval",
        hue="method",
        hue_order=hue_list,
        # order = [1, 2, 3, 4, 5],
        # row="method",
        col="dataset",
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
    axs[0].set_ylim(.0, max_*1.1)
    axs[0].set_title("Train")
    axs[1].set_title("Test")

    (fg.despine(bottom=True)
     .set_xticklabels(['Begin', 'Early', 'Mid', 'Late', 'End'])
     .set_axis_labels("", "Boundary preserving property")
     )
    fg.fig.suptitle(dataset)

    #%%

    fg.savefig(
        "boundary_{}.pdf".format(k),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Draw boundary neighbor plot for different datasets.")
    #
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("-s", type=int)
    # parser.add_argument('-e', type=int)
    # parser.add_argument('-p', type=int)
    # args = parser.parse_args()
    main()

