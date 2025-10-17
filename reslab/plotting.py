import matplotlib.pyplot as plt
import numpy as np


def line_with_ci(x, y, lo, hi, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.plot(x, y, marker="o", label="mean")
    plt.fill_between(x, lo, hi, alpha=0.2, label="CI")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bars_with_err(xlabels, means, los, his, ylabel, title, out_path):
    x = np.arange(len(xlabels))
    plt.figure()
    plt.bar(x, means)
    yerr = [means - los, his - means]
    plt.errorbar(x, means, yerr=yerr, fmt="none")
    plt.xticks(x, xlabels, rotation=20)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def heatmap(Z, xlabels, ylabels, xlabel, ylabel, title, out_path):
    plt.figure()
    plt.imshow(Z, aspect="auto", origin="lower")
    plt.colorbar(label="Survival")
    plt.xticks(range(len(xlabels)), xlabels)
    plt.yticks(range(len(ylabels)), ylabels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
