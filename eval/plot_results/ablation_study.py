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
    metric = args.method
    datasets = ["mnist", "fmnist", "cifar10"]
    starts = [4, 10, 40]
    ends = [24, 60, 240]
    periods = [4, 10, 40]
    k_neighbors = [10, 15, 20]

    if metric == "nn" or metric == "bound":
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
                content_path = "E:\\DVI_exp_data\\TemporalExp\\resnet18_{}".format(dataset)
                for epoch in [1, 2, 3, 4, 7]:
                    eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train = round(eval["{}_train_{}".format(metric, k)], 3)
                    nn_test = round(eval["{}_test_{}".format(metric, k)], 3)
                    if epoch>5:
                        i = 5
                    else:
                        i = epoch
                    if len(data) == 0:
                        data = np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(k), "{}".format(str(i)), nn_train]])
                    else:
                        data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(k), "{}".format(str(i)), nn_train]])), axis=0)
                    data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-Test", "{}".format(k), "{}".format(str(i)), nn_test]])), axis=0)

                # parametric umap
                content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
                for epoch in range(start, end, p):
                    eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap_T.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train = round(eval["{}_train_{}".format(metric, k)], 3)
                    nn_test = round(eval["{}_test_{}".format(metric, k)], 3)

                    data = np.concatenate((data, np.array([[dataset, "parametricUmap", "Train", "pU-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])), axis=0)
                    data = np.concatenate((data, np.array([[dataset, "parametricUmap", "Test", "pU-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)

                # parametric umap + Attention
                content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
                for epoch in range(start, end, p):
                    eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap_T_A.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train = round(eval["{}_train_{}".format(metric, k)], 3)
                    nn_test = round(eval["{}_test_{}".format(metric, k)], 3)

                    data = np.concatenate((data, np.array([[dataset, "parametricUmap_A", "Train", "A-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])), axis=0)
                    data = np.concatenate((data, np.array([[dataset, "parametricUmap_A", "Test", "A-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)

                # parametric umap + boundary complex
                content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
                for epoch in range(start, end, p):
                    eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_T.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train = round(eval["{}_train_{}".format(metric, k)], 3)
                    nn_test = round(eval["{}_test_{}".format(metric, k)], 3)

                    data = np.concatenate((data, np.array([[dataset, "BC", "Train", "BC-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])), axis=0)
                    data = np.concatenate((data, np.array([[dataset, "BC", "Test", "BC-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)

                # parametric umap + temporal
                content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
                for epoch in range(start, end, p):
                    eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap_step2.json")
                    with open(eval_path, "r") as f:
                        eval = json.load(f)
                    nn_train = round(eval["{}_train_{}".format(metric, k)], 3)
                    nn_test = round(eval["{}_test_{}".format(metric, k)], 3)

                    data = np.concatenate((data, np.array([[dataset, "Temporal", "Train", "T-Train", "{}".format(k), "{}".format(str(epoch//p)), nn_train]])), axis=0)
                    data = np.concatenate((data, np.array([[dataset, "Temporal", "Test", "T-Test", "{}".format(k), "{}".format(str(epoch//p)), nn_test]])), axis=0)
                df_tmp = pd.DataFrame(data, columns=col)
                df = df.append(df_tmp, ignore_index=True)
                df[["period"]] = df[["period"]].astype(int)
                df[["k"]] = df[["k"]].astype(int)
                df[["eval"]] = df[["eval"]].astype(float)
    else:
        col = np.array(["dataset", "method", "type", "hue", "period", "eval"])
        df = pd.DataFrame({}, columns=col)
        for i in range(3):
            dataset = datasets[i]
            start = starts[i]
            end = ends[i]
            p = periods[i]

            data = np.array([])
            # load data from evaluation.json
            content_path = "E:\\DVI_exp_data\\TemporalExp\\resnet18_{}".format(dataset)
            for epoch in [1, 2, 3, 4, 7]:
                eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_step2.json")
                with open(eval_path, "r") as f:
                    eval = json.load(f)
                nn_train = round(eval["{}_train".format(metric)], 3)
                nn_test = round(eval["{}_test".format(metric)], 3)
                if epoch > 5:
                    i = 5
                else:
                    i = epoch

                if len(data) == 0:
                    data = np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(str(i)), nn_train]])
                else:
                    data = np.concatenate((data, np.array([[dataset, "DVI", "Train", "DVI-Train", "{}".format(str(i)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "DVI", "Test", "DVI-Test", "{}".format(str(i)), nn_test]])), axis=0)

            # parametric umap
            content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
            for epoch in range(start, end, p):
                eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap_T.json")
                with open(eval_path, "r") as f:
                    eval = json.load(f)
                nn_train = round(eval["{}_train".format(metric)], 3)
                nn_test = round(eval["{}_test".format(metric)], 3)

                data = np.concatenate((data, np.array([[dataset, "parametricUmap", "Train", "pU-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "parametricUmap", "Test", "pU-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

            # parametric umap + Attention
            content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
            for epoch in range(start, end, p):
                eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap_T_A.json")
                with open(eval_path, "r") as f:
                    eval = json.load(f)
                nn_train = round(eval["{}_train".format(metric)], 3)
                nn_test = round(eval["{}_test".format(metric)], 3)

                data = np.concatenate((data, np.array([[dataset, "parametricUmap_A", "Train", "A-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "parametricUmap_A", "Test", "A-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

            # parametric umap + boundary complex
            content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
            for epoch in range(start, end, p):
                eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_T.json")
                with open(eval_path, "r") as f:
                    eval = json.load(f)
                nn_train = round(eval["{}_train".format(metric)], 3)
                nn_test = round(eval["{}_test".format(metric)], 3)

                data = np.concatenate((data, np.array([[dataset, "BC", "Train", "BC-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "BC", "Test", "BC-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

            # parametric umap + temporal
            content_path = "E:\\DVI_exp_data\\resnet18_{}".format(dataset)
            for epoch in range(start, end, p):
                eval_path = os.path.join(content_path, "Model", "Epoch_{}".format(epoch), "evaluation_parametricUmap_step2.json")
                with open(eval_path, "r") as f:
                    eval = json.load(f)
                nn_train = round(eval["{}_train".format(metric)], 3)
                nn_test = round(eval["{}_test".format(metric)], 3)

                data = np.concatenate((data, np.array([[dataset, "Temporal", "Train", "T-Train", "{}".format(str(epoch//p)), nn_train]])), axis=0)
                data = np.concatenate((data, np.array([[dataset, "Temporal", "Test", "T-Test", "{}".format(str(epoch//p)), nn_test]])), axis=0)

            df_tmp = pd.DataFrame(data, columns=col)
            df = df.append(df_tmp, ignore_index=True)
            df[["period"]] = df[["period"]].astype(int)
            df[["eval"]] = df[["eval"]].astype(float)

    df.to_excel("pU_{}.xlsx".format(metric))
    if metric == "nn" or metric == "bound":
        for k in k_neighbors:
            df_tmp = df[df["k"] == k]
            pal20c = sns.color_palette('tab20c', 20)
            sns.set_theme(style="whitegrid", palette=pal20c)
            hue_dict = {
                "DVI-Train": pal20c[0],
                "pU-Train": pal20c[16],
                "A-Train": pal20c[4],
                "BC-Train": pal20c[12],
                "T-Train": pal20c[8],
                "DVI-Test": pal20c[3],
                "pU-Test": pal20c[19],
                "A-Test": pal20c[7],
                "BC-Test": pal20c[15],
                "T-Test": pal20c[11],

            }
            sns.palplot([hue_dict[i] for i in hue_dict.keys()])

            axes = {'labelsize': 9,
                    'titlesize': 9,}
            mpl.rc('axes', **axes)
            mpl.rcParams['xtick.labelsize'] = 9

            hue_list = ["DVI-Train", "DVI-Test", "pU-Train", "pU-Test", "BC-Train", "BC-Test", "A-Train", "A-Test", "T-Train", "T-Test"]

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
                data=df_tmp,
                kind="bar",
                palette=[hue_dict[i] for i in hue_list],
                legend=True
            )
            sns.move_legend(fg, "lower center", bbox_to_anchor=(.42, 0.92), ncol=5, title=None, frameon=False)
            mpl.pyplot.setp(fg._legend.get_texts(), fontsize='9')

            axs = fg.axes[0]
            max_ = df_tmp["eval"].max()
            # min_ = df["eval"].min()
            axs[0].set_ylim(0., max_*1.1)
            axs[0].set_title("MNIST")
            axs[1].set_title("FMNIST")
            axs[2].set_title("CIFAR-10")

            (fg.despine(bottom=False, right=False, left=False, top=False)
             .set_xticklabels(['Begin', 'Early', 'Mid', 'Late', 'End'])
             .set_axis_labels("", "")
             )

            #%%
            fg.savefig(
                "pU_{}_{}.png".format(metric, k),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.0,
                transparent=True,
            )
    else:
        pal20c = sns.color_palette('tab20c', 20)
        # sns.palplot(pal20c)
        sns.set_theme(style="whitegrid", palette=pal20c)
        hue_dict = {
            "DVI-Train": pal20c[0],
            "pU-Train": pal20c[16],
            "A-Train": pal20c[4],
            "BC-Train": pal20c[12],
            "T-Train": pal20c[8],
            "DVI-Test": pal20c[3],
            "pU-Test": pal20c[19],
            "A-Test": pal20c[7],
            "BC-Test": pal20c[15],
            "T-Test": pal20c[11],

        }
        sns.palplot([hue_dict[i] for i in hue_dict.keys()])

        axes = {'labelsize': 9,
                'titlesize': 9,}
        mpl.rc('axes', **axes)
        mpl.rcParams['xtick.labelsize'] = 9

        hue_list = ["DVI-Train", "DVI-Test", "pU-Train", "pU-Test", "BC-Train", "BC-Test", "A-Train", "A-Test", "T-Train", "T-Test"]

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
         .set_axis_labels("", "")
         )
        # fg.fig.suptitle("NN preserving property")

        #%%
        fg.savefig(
            "pU_{}.png".format(metric),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.0,
            transparent=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw plot for different metrics.")
    #
    # parser.add_argument("--dataset", type=str)
    # parser.add_argument("-s", type=int)
    # parser.add_argument('-e', type=int)
    # parser.add_argument('-p', type=int)
    parser.add_argument("--method", '-m', type=str, choices=["nn", "bound", "inv_acc", "inv_conf"])
    args = parser.parse_args()
    main(args)

