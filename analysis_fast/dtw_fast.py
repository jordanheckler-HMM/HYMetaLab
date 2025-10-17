from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def dtw_distance(a, b):
    # simple DP for 1D arrays
    na = len(a)
    nb = len(b)
    D = np.full((na + 1, nb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return np.sqrt(D[na, nb])


def reduce_epoch(X, mode="pca1"):
    # X: T x channels
    if mode == "pca1":
        pca = PCA(n_components=1)
        return pca.fit_transform(X).ravel()
    elif mode == "avg_channels":
        return X.mean(axis=1)
    else:
        return X.mean(axis=1)


def build_epoch_matrix(df, cols=None):
    if cols is None:
        cols = ["CCI", "Meaning", "Coherence", "Rc", "epsilon"]
    groups = [g[1][cols].to_numpy(dtype=float) for g in df.groupby("epoch")]
    return groups


def dtw_epoch_matrix(groups, mode="pca1"):
    n = len(groups)
    mat = np.zeros((n, n))
    reps = [reduce_epoch(g, mode=mode) for g in groups]
    for i in range(n):
        for j in range(i, n):
            d = dtw_distance(reps[i], reps[j])
            mat[i, j] = d
            mat[j, i] = d
    return mat


def save_heatmap(mat, path, dpi=110):
    plt.figure(figsize=(5, 4), dpi=dpi)
    plt.imshow(mat, origin="lower", cmap="viridis", interpolation="nearest")
    plt.colorbar()
    plt.xlabel("epoch")
    plt.ylabel("epoch")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
