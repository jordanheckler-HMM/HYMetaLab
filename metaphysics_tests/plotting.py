import matplotlib.pyplot as plt


def save_rec_plot(R, path, dpi=220):
    plt.figure(figsize=(6, 6))
    plt.imshow(R, cmap="binary", origin="lower")
    plt.title("Recurrence Plot")
    plt.xlabel("t")
    plt.ylabel("t")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def save_heatmap(M, path, xlabel="x", ylabel="y", title="Heatmap", dpi=220):
    plt.figure(figsize=(6, 5))
    plt.imshow(M, origin="lower", aspect="auto")
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def timeseries_example(ts, path, labels=None, dpi=220):
    plt.figure(figsize=(8, 3))
    for i, s in enumerate(ts):
        plt.plot(s, label=(labels[i] if labels else f"s{i}"))
    if labels:
        plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()
