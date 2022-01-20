# %%
from comet_ml import API
from comet_ml.query import Tag
from comet_ml.query import Parameter

import matplotlib.pyplot as plt
import numpy as np


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


def plot_line_graph():
    """plot 3domain result of plot tag for comet_ml"""

    api = API(api_key="TawRAwNJiQjPaSMvBAwk4L4pF")
    tag = Tag("plot")
    experiment = api.query("kazukiomi", "feature-extract", tag)

    dataset_list = ["UCF101", "Kinetics", "HMDB51"]
    plot_dict = {"UCF101": [], "Kinetics": [], "HMDB51": []}

    plot_metric = ["train_accuracy", "val_accuracy", "train_loss", "val_loss"]
    plot_id = 3
    plot_metric = plot_metric[plot_id]

    fig, ax = plt.subplots()
    left = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000,
                     8000, 9000, 10000, 11000, 12000, 13000, 14000])

    for dataset in dataset_list:
        acc_list = experiment[0].get_metrics(plot_metric + "_" + dataset)
        for metric in acc_list:
            plot_dict[dataset].append(float(metric.get("metricValue")))
        # plt.plot(left, plot_dict[dataset], label=dataset)
        plt.plot(left, plot_dict[dataset])

    ax.set_xlabel("iteration", size=18)
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
    if plot_id == 0:
        ax.set_ylabel("train loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    if plot_id == 1:
        ax.set_ylabel("val loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    if plot_id == 2:
        ax.set_ylabel("train top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    if plot_id == 3:
        ax.set_ylabel("val top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

    plt.savefig("plot/domain_3/" + plot_metric + ".pdf")


def plot_line_graph2():
    """plot 1domain result of plot2 tag for comet_ml"""

    api = API(api_key="TawRAwNJiQjPaSMvBAwk4L4pF")
    tag = Tag("plot2")
    experiment = api.query("kazukiomi", "feature-extract", tag)

    dataset_list = ["UCF101", "Kinetics", "HMDB51"]
    dataset_list2 = ["UCF101", "Kinetics400", "HMDB51"]
    plot_dict = {"UCF101": [], "Kinetics": [], "HMDB51": []}

    plot_metric = ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]
    plot_id = 0
    plot_metric = plot_metric[plot_id]

    fig, ax = plt.subplots()
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
    ax.set_xticks([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000])
    if plot_id == 0:
        ax.set_ylabel("train loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        plt.legend(fontsize=24)
    if plot_id == 1:
        ax.set_ylabel("val loss", size=24)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    if plot_id == 2:
        ax.set_ylabel("train top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    if plot_id == 3:
        ax.set_ylabel("val top-1 accuracy (%)", size=24)
        ax.set_yticks([0, 20, 40, 60, 80, 100])

    plt.savefig("plot/domain_1/" + plot_metric + ".pdf")


if __name__ == "__main__":
    plot_line_graph2()
    # plot_bar_graph()
