# %%
from comet_ml import API
from comet_ml.query import Tag
from comet_ml.query import Parameter

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    api = API(api_key="TawRAwNJiQjPaSMvBAwk4L4pF")
    tag = Tag("ex_adp_mode")
    experiment = api.query("kazukiomi", "feature-extract", tag)
    print(experiment)

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
