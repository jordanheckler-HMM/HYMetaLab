import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


def build_state_matrix(df, cols):
    X = df[cols].astype(float).values
    return X


def pairwise_distances(X, metric="euclidean"):
    D = squareform(pdist(X, metric=metric))
    return D


def recurrence_matrix(
    X, tau="percentile", tau_percentile=10, tau_fixed=None, metric="euclidean"
):
    D = pairwise_distances(X, metric=metric)
    if tau == "percentile":
        thresh = np.percentile(D, tau_percentile)
    else:
        thresh = tau_fixed if tau_fixed is not None else np.median(D)
    R = (D <= thresh).astype(int)
    return R, thresh


def recurrence_plot(R, outpath=None, dpi=220, title="Recurrence Plot"):
    plt.figure(figsize=(6, 6))
    plt.imshow(R, cmap="binary", origin="lower")
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel("time")
    if outpath:
        plt.savefig(outpath, dpi=dpi)
    plt.close()


def rqa_metrics(R, min_diag=2, min_vert=2):
    n = R.shape[0]
    RR = R.sum() / (n * n)
    # diag line statistics
    diag_lines = []
    for k in range(-n + 1, n):
        diag = np.diag(R, k=k)
        # find runs of 1s
        runs = _runs_lengths(diag)
        diag_lines.extend([r for r in runs if r >= min_diag])
    DET = sum(diag_lines) / R.sum() if R.sum() > 0 else 0.0
    Lmax = max(diag_lines) if diag_lines else 0
    # entropy of diag lines
    if diag_lines:
        p, _ = np.histogram(
            diag_lines, bins=range(1, max(diag_lines) + 2), density=True
        )
        ENTR = -np.sum([pi * np.log(pi) for pi in p if pi > 0])
    else:
        ENTR = 0.0
    # vertical lines -> laminarity
    vert_lines = []
    for col in range(n):
        runs = _runs_lengths(R[:, col])
        vert_lines.extend([r for r in runs if r >= min_vert])
    LAM = sum(vert_lines) / R.sum() if R.sum() > 0 else 0.0
    TT = np.mean(vert_lines) if vert_lines else 0.0
    return {"RR": RR, "DET": DET, "LAM": LAM, "Lmax": int(Lmax), "ENTR": ENTR, "TT": TT}


def _runs_lengths(arr):
    runs = []
    count = 0
    for v in arr:
        if v == 1:
            count += 1
        else:
            if count > 0:
                runs.append(count)
                count = 0
    if count > 0:
        runs.append(count)
    return runs
