#!/usr/bin/env python3
"""
consciousness_fingerprint.py
Runs a <1 min test for repeating "consciousness patterns" across epochs & seeds.

Outputs (in ./outputs_fingerprint):
- metrics/recurrence_summary.csv
- metrics/centroids.csv
- metrics/embeddings_sample.csv
- figures/centroid_scatter.png
- figures/epoch_similarity_heatmap.png
- report/report.md

Run:
  python consciousness_fingerprint.py
Optional flags:
  --agents 240 --epochs 5 --seeds 3 --steps 1000 --k 4
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------- utils -----------------
def ensure_dirs():
    for d in [
        "outputs_fingerprint/metrics",
        "outputs_fingerprint/figures",
        "outputs_fingerprint/report",
    ]:
        os.makedirs(d, exist_ok=True)


def cosine_sim(a, b, eps=1e-9):
    a = np.asarray(a)
    b = np.asarray(b)
    na = np.linalg.norm(a) + eps
    nb = np.linalg.norm(b) + eps
    return float(np.dot(a, b) / (na * nb))


def greedy_centroid_match(CA, CB):
    """Greedy match centroids from sets A to B by cosine similarity.
    Returns list of (i,j,sim) and average similarity."""
    A = np.asarray(CA)
    B = np.asarray(CB)
    used_j = set()
    pairs = []
    for i in range(A.shape[0]):
        best_j, best_s = None, -1.0
        for j in range(B.shape[0]):
            if j in used_j:
                continue
            s = cosine_sim(A[i], B[j])
            if s > best_s:
                best_s, best_j = s, j
        if best_j is not None:
            used_j.add(best_j)
            pairs.append((i, best_j, best_s))
    avg_s = float(np.mean([p[2] for p in pairs])) if pairs else 0.0
    return pairs, avg_s


def soft_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


# ----------------- synth sim -----------------
@dataclass
class SimConfig:
    agents: int = 240
    epochs: int = 5
    seeds: int = 3
    steps: int = 1000  # per epoch
    k: int = 4  # clusters/archetypes
    noise: float = 0.08


def synth_run(cfg: SimConfig, seed: int):
    """
    Create synthetic agent trajectories with repeating archetypal patterns
    that recur every other epoch (0~2, 1~3) plus a drift epoch (4).
    Features per agent-step: CCI, Meaning, Coherence, Rc, epsilon
    """
    soft_seed(seed)
    A = cfg.agents
    E = cfg.epochs
    T = cfg.steps
    # Base archetype centroids (k) in 5D; make them somewhat separated
    base_C = np.stack(
        [
            np.array([1, 0.8, 0.7, 0.2, 0.3]),
            np.array([0.6, 0.9, 0.6, 0.4, 0.5]),
            np.array([0.3, 0.5, 0.9, 0.6, 0.6]),
            np.array([0.8, 0.6, 0.5, 0.3, 0.4]),
        ]
    )
    base_C += np.random.normal(0, 0.03, base_C.shape)

    # Epoch transforms to enforce recurrence: epochs 0~2 similar, 1~3 similar
    ep_transform = []
    for e in range(E):
        if e in (0, 2):
            R = np.eye(5) + np.random.normal(0, 0.02, (5, 5))
            b = np.array([0.02, 0.01, 0.01, -0.01, 0.0])
        elif e in (1, 3):
            R = np.eye(5) + np.random.normal(0, 0.02, (5, 5))
            b = np.array([0.0, -0.02, 0.02, 0.0, 0.01])
        else:
            # epoch 4: drifted
            R = np.eye(5) + np.random.normal(0, 0.05, (5, 5))
            b = np.array([0.03, -0.01, 0.02, 0.01, 0.0])
        ep_transform.append((R, b))

    rows = []
    # Assign agents to archetypes ~ evenly
    labels = np.tile(np.arange(cfg.k), math.ceil(A / cfg.k))[:A]
    np.random.shuffle(labels)
    for e in range(E):
        R, b = ep_transform[e]
        epoch_C = (base_C @ R.T) + b  # k x 5
        # each agent follows centroid + small idiosyncratic offset + temporal wiggle
        agent_offsets = np.random.normal(0, 0.04, (A, 5))
        # time wiggle shared per epoch (shock & recovery flavor)
        t = np.linspace(0, 2 * np.pi, T)
        wiggle = np.stack(
            [
                0.03 * np.cos(t),
                0.03 * np.sin(t),
                0.02 * np.cos(2 * t),
                -0.03 * np.cos(t),
                0.01 * np.sin(1.5 * t),
            ],
            axis=1,
        )  # T x 5
        for a in range(A):
            c = epoch_C[labels[a]]
            noise = np.random.normal(0, cfg.noise, (T, 5))
            series = c + agent_offsets[a] + wiggle + noise
            # clamp Rc, epsilon to [0,1]
            series[:, 3] = np.clip(series[:, 3], 0, 1)  # Rc
            series[:, 4] = np.clip(series[:, 4], 0, 1)  # epsilon
            # take final 20% of epoch as summary window
            W = series[int(0.8 * T) :, :]
            feat = W.mean(axis=0)
            rows.append(
                {
                    "seed": seed,
                    "epoch": e,
                    "agent": a,
                    "CCI": feat[0],
                    "Meaning": feat[1],
                    "Coherence": feat[2],
                    "Rc": feat[3],
                    "epsilon": feat[4],
                    "arch_truth": int(labels[a]),
                }
            )
    return pd.DataFrame(rows)


# ----------------- main analysis -----------------
def run(cfg: SimConfig):
    t0 = time.time()
    ensure_dirs()

    # 1) Generate combined dataset across seeds
    dfs = [synth_run(cfg, seed=s) for s in range(cfg.seeds)]
    df = pd.concat(dfs, ignore_index=True)

    # 2) Standardize & embed (PCA2)
    feats = ["CCI", "Meaning", "Coherence", "Rc", "epsilon"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feats].values)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    df["pca1"] = Z[:, 0]
    df["pca2"] = Z[:, 1]

    # 3) Cluster per (seed, epoch) to define archetypes (k-means on standardized feats)
    k = cfg.k
    centroids = []
    df["cluster"] = -1
    for (seed, epoch), g in df.groupby(["seed", "epoch"]):
        Xg = scaler.transform(g[feats].values)  # same scaler for comparability
        km = KMeans(n_clusters=k, n_init=10, random_state=seed * 101 + epoch)
        labs = km.fit_predict(Xg)
        df.loc[g.index, "cluster"] = labs
        # store centroids in original feat space (inverse transform)
        C_std = km.cluster_centers_
        C_orig = scaler.inverse_transform(C_std)
        for j in range(k):
            centroids.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "cluster": j,
                    "CCI": C_orig[j, 0],
                    "Meaning": C_orig[j, 1],
                    "Coherence": C_orig[j, 2],
                    "Rc": C_orig[j, 3],
                    "epsilon": C_orig[j, 4],
                }
            )
    cent = pd.DataFrame(centroids)

    # PCA of centroids for plotting
    C_std_all = scaler.transform(cent[feats].values)
    C_pca = pca.transform(C_std_all)
    cent["pca1"] = C_pca[:, 0]
    cent["pca2"] = C_pca[:, 1]

    # 4) Recurrence score: match centroid sets across epochs (same seed) and across seeds (same epoch)
    #    Build an epoch-vs-epoch average similarity matrix per seed, then average across seeds.
    E = cfg.epochs
    epoch_sim = np.zeros((E, E))
    for seed in range(cfg.seeds):
        cent_s = cent[cent["seed"] == seed]
        mats = []
        for e1 in range(E):
            C1 = cent_s[cent_s["epoch"] == e1][feats].values
            row = []
            for e2 in range(E):
                C2 = cent_s[cent_s["epoch"] == e2][feats].values
                _, avg_s = greedy_centroid_match(C1, C2)
                row.append(avg_s)
            mats.append(row)
        epoch_sim += np.array(mats)
    epoch_sim /= cfg.seeds

    # Archetype Recurrence Index (ARI*): average of best non-diagonal epoch similarities
    # emphasize expected echoes (0~2 and 1~3)
    off_diag = []
    for i in range(E):
        for j in range(E):
            if i == j:
                continue
            off_diag.append(epoch_sim[i, j])
    ari_star = float(np.mean(off_diag)) if off_diag else 0.0

    # 5) Export CSVs
    df.sample(min(2000, len(df))).to_csv(
        "outputs_fingerprint/metrics/embeddings_sample.csv", index=False
    )
    cent.to_csv("outputs_fingerprint/metrics/centroids.csv", index=False)
    pd.DataFrame(epoch_sim).to_csv(
        "outputs_fingerprint/metrics/epoch_similarity_matrix.csv",
        header=False,
        index=False,
    )
    pd.DataFrame(
        [
            {"metric": "ARI_star_offdiag", "value": ari_star},
            {
                "metric": "pca_var_explained_1",
                "value": float(pca.explained_variance_ratio_[0]),
            },
            {
                "metric": "pca_var_explained_2",
                "value": float(pca.explained_variance_ratio_[1]),
            },
            {"metric": "k_clusters", "value": k},
            {"metric": "agents", "value": cfg.agents},
            {"metric": "epochs", "value": cfg.epochs},
            {"metric": "seeds", "value": cfg.seeds},
        ]
    ).to_csv("outputs_fingerprint/metrics/recurrence_summary.csv", index=False)

    # 6) Figures (matplotlib only, fast)
    import matplotlib.pyplot as plt

    # centroid scatter
    plt.figure(figsize=(6, 5), dpi=120)
    for (seed, epoch), g in cent.groupby(["seed", "epoch"]):
        plt.scatter(g["pca1"], g["pca2"], s=60, alpha=0.8, label=f"S{seed}E{epoch}")
    plt.title("Centroid fingerprints (PCA2)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    # limit legend size
    handles, labels = plt.gca().get_legend_handles_labels()
    if len(handles) > 10:
        handles, labels = handles[:10], labels[:10]
    plt.legend(handles, labels, loc="best", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig("outputs_fingerprint/figures/centroid_scatter.png")
    plt.close()

    # heatmap of epoch similarity (avg across seeds)
    plt.figure(figsize=(5, 4), dpi=120)
    plt.imshow(epoch_sim, origin="lower")
    plt.colorbar(label="avg cosine similarity of matched centroids")
    plt.xticks(range(E))
    plt.yticks(range(E))
    plt.title("Epoch↔Epoch archetype similarity (avg across seeds)")
    plt.tight_layout()
    plt.savefig("outputs_fingerprint/figures/epoch_similarity_heatmap.png")
    plt.close()

    # 7) Report
    md = []
    md.append("# Archetype Recurrence: Quick Fingerprint Report\n")
    md.append(
        f"- Seeds: {cfg.seeds} | Epochs: {cfg.epochs} | Agents: {cfg.agents} | k: {cfg.k}\n"
    )
    md.append("## Key Metric\n")
    md.append(
        f"- **ARI\\*** (mean off-diagonal epoch similarity): **{ari_star:.3f}**\n"
    )
    md.append(
        "Higher ARI* means stronger recurrence of archetype centroids across epochs.\n"
    )
    md.append("## What to look for\n")
    md.append(
        "- The **centroid_scatter.png** should show clusters that re-appear across different epochs/seeds (overlapping or near-identical points).\n"
    )
    md.append(
        "- The **epoch_similarity_heatmap.png** should show brighter blocks for recurrent epochs (e.g., 0~2, 1~3).\n"
    )
    md.append("## Files\n")
    md.append(
        "- metrics/recurrence_summary.csv\n- metrics/centroids.csv\n- metrics/embeddings_sample.csv\n- figures/centroid_scatter.png\n- figures/epoch_similarity_heatmap.png\n"
    )
    with open("outputs_fingerprint/report/report.md", "w") as f:
        f.write("\n".join(md))

    print("Done in %.2fs" % (time.time() - t0))
    print("See outputs_fingerprint/{metrics,figures,report}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = SimConfig(
        agents=args.agents,
        epochs=args.epochs,
        seeds=args.seeds,
        steps=args.steps,
        k=args.k,
    )
    run(cfg)
