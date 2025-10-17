from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def build_state(df, cols=None):
    if cols is None:
        cols = ["CCI", "Meaning", "Coherence", "Rc", "epsilon"]
    X = df[cols].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X)
    return X


def recurrence_plot(X, tau_percentile=10):
    d = pairwise_distances(X, metric="euclidean")
    tau = np.percentile(d, tau_percentile)
    R = (d <= tau).astype(int)
    return R, tau


def _line_statistics(R):
    # R is binary square matrix
    N = R.shape[0]
    # Diagonal lines: count runs on each k-diagonal (offsets)
    diags = []
    for offset in range(-N + 1, N):
        diag = np.diag(R, k=offset)
        if diag.size == 0:
            continue
        # runs of ones
        runs = np.diff(np.concatenate(([0], diag, [0])))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        lengths = ends - starts
        diags.extend(lengths.tolist())
    diags = np.array(diags, dtype=int) if len(diags) else np.array([], dtype=int)

    # Vertical lines: runs in columns
    verts = []
    for col in range(N):
        runs = np.diff(np.concatenate(([0], R[:, col], [0])))
        starts = np.where(runs == 1)[0]
        ends = np.where(runs == -1)[0]
        lengths = ends - starts
        verts.extend(lengths.tolist())
    verts = np.array(verts, dtype=int) if len(verts) else np.array([], dtype=int)

    return diags, verts


def rqa_metrics(R, min_diag=2, min_vert=2):
    N = R.shape[0]
    total_points = N * N
    RR = R.sum() / total_points

    diags, verts = _line_statistics(R)

    diag_lines = diags[diags >= min_diag] if diags.size else np.array([])
    vert_lines = verts[verts >= min_vert] if verts.size else np.array([])

    DET = diag_lines.sum() / diags.sum() if diags.size and diags.sum() > 0 else 0.0
    LAM = vert_lines.sum() / verts.sum() if verts.size and verts.sum() > 0 else 0.0
    Lmax = diag_lines.max() if diag_lines.size else 0
    # entropy of diagonal line lengths
    if diag_lines.size:
        vals, counts = np.unique(diag_lines, return_counts=True)
        probs = counts / counts.sum()
        ENTR = -np.sum(probs * np.log(probs))
    else:
        ENTR = 0.0
    TT = vert_lines.mean() if vert_lines.size else 0.0

    return dict(
        RR=float(RR),
        DET=float(DET),
        LAM=float(LAM),
        Lmax=int(Lmax),
        ENTR=float(ENTR),
        TT=float(TT),
    )


def save_rp(R, path, dpi=110):
    plt.figure(figsize=(4, 4), dpi=dpi)
    plt.imshow(R, cmap="Greys", origin="lower", interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def analyze(df, tau_percentile=10, outdir="outputs_fast/figures"):
    outdir = Path(outdir)
    X = build_state(df)
    R, tau = recurrence_plot(X, tau_percentile=tau_percentile)
    metrics = rqa_metrics(R)
    save_rp(R, outdir / "recurrence_plot_GLOBAL.png")
    return metrics, R, tau


def analyze_epochs(df, tau_percentile=10, outdir="outputs_fast/figures"):
    outdir = Path(outdir)
    epochs = {}
    epoch_metrics = []
    for e, g in df.groupby("epoch"):
        X = build_state(g)
        R, tau = recurrence_plot(X, tau_percentile=tau_percentile)
        metrics = rqa_metrics(R)
        save_rp(R, outdir / f"recurrence_plot_EPOCH_{int(e)}.png")
        epoch_metrics.append(dict(epoch=int(e), tau=float(tau), **metrics))
    return pd.DataFrame(epoch_metrics)
