import os

import matplotlib.pyplot as plt
import numpy as np


def plot_lesion_comparison(on_values, off_values, out_png):
    means = [
        np.mean(on_values) if on_values else 0,
        np.mean(off_values) if off_values else 0,
    ]
    sems = [
        (
            np.std(on_values, ddof=1) / np.sqrt(len(on_values))
            if len(on_values) > 1
            else 0
        ),
        (
            np.std(off_values, ddof=1) / np.sqrt(len(off_values))
            if len(off_values) > 1
            else 0
        ),
    ]
    labels = ["workspace_on", "workspace_off"]
    x = np.arange(len(labels))
    plt.figure()
    plt.bar(x, means, yerr=sems, capsize=5)
    plt.xticks(x, labels)
    plt.ylabel("Average reward")
    plt.title("Lesion: workspace on vs off")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    print("Saved lesion comparison plot to", out_png)
