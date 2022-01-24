# %%
from comet_ml import API
from comet_ml.query import Tag
from comet_ml.query import Parameter

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np


import argparse
import configparser


def plot_bar_graph():
    api = API(api_key="TawRAwNJiQjPaSMvBAwk4L4pF")
    tag = Tag("ex_adp_mode")
    experiment = api.query("kazukiomi", "feature-extract", tag)
    dataset_list = ["UCF101", "Kinetics", "HMDB51"]
    plot_dict = {"UCF101": [], "Kinetics": [], "HMDB51": []}
    for i, ex in enumerate(experiment):
        if i > 1:
            break
        for dataset in dataset_list:
            try:
                adp_mode = ex.get_parameters_summary("mode")["valueMax"]
                # print(adp_mode)
                metric_dict = ex.get_metrics("val_accuracy_" + dataset)[-1]
                data = metric_dict["metricName"][13:]
                acc = float(metric_dict["metricValue"])
                print(f"{data} : {acc} : {adp_mode}")
                plot_dict[data].append(acc)
            except IndexError:
                pass
    print(plot_dict)

    labels = ["(2+1)D conv", "Frame-Wise 2D conv"]
    mergin = 0.4
    total_width = 1 - mergin
    x = np.array([1, 2])
    for i, list in enumerate(dataset_list):
        pos = x - total_width * (1 - (2 * i + 1) / len(dataset_list)) / 2
        plt.bar(pos, plot_dict[list],
                width=total_width / len(dataset_list), label=dataset_list[i])
    plt.xticks(x, labels)
    plt.legend()
    plt.savefig("fig.png")


def plot_line_graph(plot_metric):
    """plot 3domain result of plot tag for comet_ml"""

    api = API(api_key="TawRAwNJiQjPaSMvBAwk4L4pF")
    tag = Tag("plot")
    experiment = api.query("kazukiomi", "feature-extract", tag)

    dataset_list = ["UCF101", "Kinetics", "HMDB51"]
    plot_dict = {"UCF101": [], "Kinetics": [], "HMDB51": []}

    fig, ax = plt.subplots()
    ax.set_xlim(0, 15000)
    fig.subplots_adjust(bottom=0.2, left=0.2)
    left = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000,
                     8000, 9000, 10000, 11000, 12000, 13000, 14000])

    for dataset in dataset_list:
        acc_list = experiment[0].get_metrics(plot_metric + "_" + dataset)
        for metric in acc_list:
            plot_dict[dataset].append(float(metric.get("metricValue")))
        # plt.plot(left, plot_dict[dataset], label=dataset)
        plt.plot(left, plot_dict[dataset])

    ax.set_xlabel("iteration", size=24)
    ax.set_xticks([0, 5000, 10000])
    if plot_metric == "train_loss":
        ax.set_ylabel("train loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_ylim(0, 6)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    if plot_metric == "val_loss":
        ax.set_ylabel("val loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_ylim(0, 6)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    if plot_metric == "train_accuracy":
        ax.set_ylabel("train top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(10))
    if plot_metric == "val_accuracy":
        ax.set_ylabel("val top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(10))
    # plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1000))

    ax.grid(which="both", axis="x", color="black",
            alpha=0.7, linestyle="--", linewidth=0.5,)
    ax.grid(which="both", axis="y", color="black",
            alpha=0.7, linestyle="--", linewidth=0.5,)
    ax.tick_params(direction="out", labelsize=24)
    plt.savefig("plot/domain_3/" + plot_metric + ".pdf")


def plot_line_graph2(plot_metric):
    """plot 1domain result of plot2 tag for comet_ml"""

    api = API(api_key="TawRAwNJiQjPaSMvBAwk4L4pF")
    tag = Tag("plot2")
    experiment = api.query("kazukiomi", "feature-extract", tag)

    dataset_list = ["UCF101", "Kinetics", "HMDB51"]
    dataset_list2 = ["UCF101", "Kinetics400", "HMDB51"]
    plot_dict = {"UCF101": [], "Kinetics": [], "HMDB51": []}

    fig, ax = plt.subplots()
    ax.set_xlim(0, 15000)
    fig.subplots_adjust(bottom=0.2, left=0.2)

    left = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000,
                     8000, 9000, 10000, 11000, 12000, 13000, 14000])

    experiment[1], experiment[2] = experiment[2], experiment[1]
    for i, ex in enumerate(experiment):
        acc_list = ex.get_metrics(plot_metric + "_" + dataset_list[i])
        for metric in acc_list:
            plot_dict[dataset_list[i]].append(float(metric.get("metricValue")))
        plt.plot(left, plot_dict[dataset_list[i]], label=dataset_list2[i])
        # plt.plot(left, plot_dict[dataset_list[i]])
        # print(plot_dict[dataset_list[i]])

    ax.set_xlabel("iteration", size=24)
    ax.set_xticks([0, 5000, 10000])
    if plot_metric == "train_loss":
        ax.set_ylabel("train loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_ylim(0, 6)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
        plt.legend(fontsize=24)
    if plot_metric == "val_loss":
        ax.set_ylabel("val loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax.set_ylim(0, 6)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(1))
    if plot_metric == "train_accuracy":
        ax.set_ylabel("train top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(10))
    if plot_metric == "val_accuracy":
        ax.set_ylabel("val top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.set_ylim(0, 100)
        plt.gca().yaxis.set_minor_locator(ticker.MultipleLocator(10))
    # plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(ticker.MultipleLocator(1000))

    ax.grid(which="both", axis="x", color="black",
            alpha=0.7, linestyle="--", linewidth=0.5,)
    ax.grid(which="both", axis="y", color="black",
            alpha=0.7, linestyle="--", linewidth=0.5,)
    ax.tick_params(direction="out", labelsize=24)
    plt.savefig("plot/domain_1/" + plot_metric + ".pdf")


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-nd", "--num_domain",
                        type=int, default=1, choices=[1, 3])
    parser.add_argument("-pm", "--plot_metric", type=str,
                        choices=["train_loss", "val_loss", "train_accuracy", "val_accuracy"])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    if args.num_domain == 1:
        plot_line_graph2(args.plot_metric)
    elif args.num_domain == 3:
        plot_line_graph(args.plot_metric)
