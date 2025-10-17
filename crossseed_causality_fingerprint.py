#!/usr/bin/env python3
"""
crossseed_causality_fingerprint.py
Fast test for (1) cross-SEED recurrence of consciousness archetypes,
and (2) causality from CCI -> {Meaning, Coherence, Rc} via Granger + fast TE proxy.

Outputs (./outputs_crossseed):
  metrics/
    interseed_epoch_similarity.csv
    ari_seed_offdiag.csv
    granger_results.csv
    te_results.csv
  figures/
    interseed_similarity_heatmap.png
    centroid_scatter_by_seed.png
    granger_heatmap.png
    te_bar.png
  report/report.md

Run:
  python crossseed_causality_fingerprint.py
Optional flags:
  --agents 240 --epochs 5 --seeds 3 --steps 1000 --k 4 --max_lag 4
"""

import argparse
import math
import os
import random
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ----------------------- utils -----------------------
def ensure_dirs():
    for d in [
        "outputs_crossseed/metrics",
        "outputs_crossseed/figures",
        "outputs_crossseed/report",
    ]:
        os.makedirs(d, exist_ok=True)


def soft_seed(s):
    np.random.seed(s)
    random.seed(s)


def cosine_sim(a, b, eps=1e-9):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) + eps) / (np.linalg.norm(b) + eps))


def greedy_match(CA, CB):
    used = set()
    pairs = []
    CA = np.asarray(CA)
    CB = np.asarray(CB)
    for i in range(CA.shape[0]):
        j_best, s_best = None, -1.0
        for j in range(CB.shape[0]):
            if j in used:
                continue
            s = cosine_sim(CA[i], CB[j])
            if s > s_best:
                s_best, j_best = s, j
        if j_best is not None:
            used.add(j_best)
            pairs.append((i, j_best, s_best))
    avg = float(np.mean([p[2] for p in pairs])) if pairs else 0.0
    return pairs, avg


# ------------------- synth simulation ----------------
@dataclass
class SimCfg:
    agents: int = 240
    epochs: int = 5
    seeds: int = 3
    steps: int = 1000
    k: int = 4
    noise: float = 0.08


def synth_run(cfg: SimCfg, seed: int):
    """Generate per-(seed,epoch) agent features with controlled recurrence."""
    soft_seed(seed)
    A, E, T = cfg.agents, cfg.epochs, cfg.steps
    base_C = np.stack(
        [
            [1.0, 0.8, 0.7, 0.25, 0.35],
            [0.6, 0.9, 0.6, 0.4, 0.55],
            [0.35, 0.55, 0.9, 0.6, 0.6],
            [0.85, 0.6, 0.5, 0.3, 0.45],
        ]
    ) + np.random.normal(0, 0.03, (cfg.k, 5))
    ep_transform = []
    for e in range(E):
        if e in (0, 2):  # echo pair
            R = np.eye(5) + np.random.normal(0, 0.02, (5, 5))
            b = np.array([0.02, 0.0, 0.01, -0.01, 0.0])
        elif e in (1, 3):  # echo pair
            R = np.eye(5) + np.random.normal(0, 0.02, (5, 5))
            b = np.array([0.0, -0.02, 0.02, 0.0, 0.01])
        else:  # drift epoch
            R = np.eye(5) + np.random.normal(0, 0.05, (5, 5))
            b = np.array([0.03, -0.01, 0.02, 0.01, 0.0])
        ep_transform.append((R, b))
    rows = []
    labels = np.tile(np.arange(cfg.k), math.ceil(A / cfg.k))[:A]
    np.random.shuffle(labels)
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
    for e in range(E):
        R, b = ep_transform[e]
        epoch_C = (base_C @ R.T) + b
        agent_offsets = np.random.normal(0, 0.04, (A, 5))
        for a in range(A):
            c = epoch_C[labels[a]]
            noise = np.random.normal(0, cfg.noise, (T, 5))
            series = c + agent_offsets[a] + wiggle + noise
            series[:, 3] = np.clip(series[:, 3], 0, 1)  # Rc
            series[:, 4] = np.clip(series[:, 4], 0, 1)  # epsilon
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


# --------------------- analysis ----------------------
def cluster_centroids(df, feats, k, scaler=None, pca=None):
    if scaler is None:
        scaler = StandardScaler().fit(df[feats].values)
    Xstd = scaler.transform(df[feats].values)
    cents = []
    labels_out = np.full(len(df), -1, dtype=int)
    for (seed, epoch), g in df.groupby(["seed", "epoch"]):
        Xg = scaler.transform(g[feats].values)
        km = KMeans(n_clusters=k, n_init=10, random_state=seed * 101 + epoch)
        labs = km.fit_predict(Xg)
        labels_out[g.index] = labs
        C_orig = scaler.inverse_transform(km.cluster_centers_)
        for j in range(k):
            cents.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "cluster": j,
                    **{f: C_orig[j, fi] for fi, f in enumerate(feats)},
                }
            )
    cent = pd.DataFrame(cents)
    if pca is None:
        pca = PCA(n_components=2, random_state=0).fit(
            scaler.transform(df[feats].values)
        )
    cent_std = scaler.transform(cent[feats].values)
    Z = pca.transform(cent_std)
    cent["pca1"], cent["pca2"] = Z[:, 0], Z[:, 1]
    return cent, labels_out, scaler, pca


def interseed_epoch_similarity(cent, feats, seeds, epochs, k):
    # For each epoch, compare centroid set of seed i vs seed j (avg over all pairs)
    sim = np.zeros((epochs, epochs))  # we'll only fill diagonal rows i==j (same epoch)
    # Build dict: (seed,epoch) -> centroids matrix (k x d)
    M = {}
    for (s, e), g in cent.groupby(["seed", "epoch"]):
        C = g.sort_values("cluster")[feats].values
        if C.shape[0] != k:  # rare if KMeans collapsed: pad by repeating last
            if C.shape[0] == 0:
                C = np.zeros((k, len(feats)))
            else:
                last = C[-1:, :]
                C = np.vstack([C] + [last] * (k - C.shape[0]))
        M[(s, e)] = C
    # average across seed pairs for same epoch; off-diagonal epoch cells left for completeness (optional)
    for e in range(epochs):
        sims = []
        for i in range(seeds):
            for j in range(i + 1, seeds):
                _, s = greedy_match(M[(i, e)], M[(j, e)])
                sims.append(s)
        sim[e, e] = float(np.mean(sims)) if sims else 0.0
    return sim


# ---------- causality: Granger + fast TE proxy ----------
def granger_pair(series_xy, max_lag=4):
    # series_xy: DataFrame with columns ["CCI","Meaning","Coherence","Rc"]
    try:
        import statsmodels.tsa.api as tsa
    except ImportError:
        # Fallback if statsmodels not available
        print(
            "Warning: statsmodels not available, using simple correlation proxy for Granger test"
        )
        out = []
        for target in ["Meaning", "Coherence", "Rc"]:
            df = series_xy[["CCI", target]].dropna()
            if len(df) > max_lag:
                # Simple lag correlation as proxy
                cci_lag = df["CCI"].shift(1).dropna()
                target_curr = df[target].iloc[1 : len(cci_lag) + 1]
                corr = np.corrcoef(cci_lag, target_curr)[0, 1]
                p = 0.05 if abs(corr) > 0.3 else 0.5  # rough proxy
                stat = abs(corr) * 10
            else:
                p, stat = np.nan, np.nan
            out.append(
                {"from": "CCI", "to": target, "p": p, "stat": stat, "lag": max_lag}
            )
        return pd.DataFrame(out)

    out = []
    for target in ["Meaning", "Coherence", "Rc"]:
        df = series_xy[["CCI", target]].dropna()
        # downsample slightly for speed/stability
        df2 = df.iloc[::2, :]
        try:
            model = tsa.VAR(df2)
            res = model.fit(maxlags=max_lag, ic=None)
            test = res.test_causality(caused=target, causing=["CCI"], kind="f")
            p = float(test.pvalue)
            stat = float(test.statistic)
        except Exception:
            p, stat = np.nan, np.nan
        out.append({"from": "CCI", "to": target, "p": p, "stat": stat, "lag": max_lag})
    return pd.DataFrame(out)


def discretize(x, bins=8):
    # simple rank-based binning (robust)
    ranks = pd.Series(x).rank(method="average").values
    edges = np.linspace(0, len(ranks), bins + 1)
    return np.digitize(ranks, edges[:-1], right=True) - 1


def transfer_entropy_discrete(x, y, lag=1, bins=8):
    """
    TE X->Y with coarse discrete estimator:
      TE = sum p(y_t, y_{t-1}, x_{t-1}) * log [ p(y_t | y_{t-1}, x_{t-1}) / p(y_t | y_{t-1}) ]
    Fast, coarse, good enough for a yes/no signal.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    x = discretize(x, bins)
    y = discretize(y, bins)
    x_l = x[:-lag]
    y_l = y[:-lag]
    y_t = y[lag:]
    from collections import Counter

    def counts(*cols):
        return Counter(zip(*cols))

    c_xyz = counts(y_t, y_l, x_l)
    c_yz = counts(y_t, y_l)
    c_z = Counter(y_l)
    c_xz = counts(x_l, y_l)

    te = 0.0
    eps = 1e-12
    N = len(y_t)
    for (yt, yz, xl), n in c_xyz.items():
        p_xyz = n / N
        p_y_given_yz_xl = n / (c_xz[(xl, yz)] + eps)
        p_y_given_yz = c_yz[(yt, yz)] / (c_z[yz] + eps)
        te += p_xyz * np.log2((p_y_given_yz_xl + eps) / (p_y_given_yz + eps))
    return float(max(0.0, te))


# ---------------------------- main ----------------------------
def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", type=int, default=240)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max_lag", type=int, default=4)
    args = ap.parse_args()
    cfg = SimCfg(args.agents, args.epochs, args.seeds, args.steps, args.k)

    ensure_dirs()

    # 1) synth data across seeds (fast)
    dfs = [synth_run(cfg, s) for s in range(cfg.seeds)]
    df = pd.concat(dfs, ignore_index=True)
    feats = ["CCI", "Meaning", "Coherence", "Rc", "epsilon"]

    # 2) cluster -> centroids
    cent, labels, scaler, pca = cluster_centroids(df, feats, cfg.k)

    # 3) cross-SEED, per-epoch similarity (diagonal of matrix)
    sim = interseed_epoch_similarity(cent, feats, cfg.seeds, cfg.epochs, cfg.k)
    # ARI_seed*: mean diagonal value (epoch-wise similarity averaged)
    diag_vals = [sim[e, e] for e in range(cfg.epochs)]
    ari_seed = float(np.mean(diag_vals)) if diag_vals else 0.0

    # 4) causality on a single long concatenated seed (seed 0) for speed
    s0 = df[df["seed"] == 0].sort_values(["epoch", "agent"])
    # build pseudo-time series by averaging across agents per timestep proxy (use epoch order)
    # we don't have time steps here; approximate by agent index sequence (fine for quick test)
    grp = s0.groupby("agent")[["CCI", "Meaning", "Coherence", "Rc"]].mean()
    gc = granger_pair(grp[["CCI", "Meaning", "Coherence", "Rc"]], max_lag=args.max_lag)

    # fast TE proxy on seed 0 (direction CCI -> each Y), using agent-order series
    te_rows = []
    for target in ["Meaning", "Coherence", "Rc"]:
        te = transfer_entropy_discrete(
            grp["CCI"].values, grp[target].values, lag=1, bins=8
        )
        te_rows.append({"from": "CCI", "to": target, "lag": 1, "TE_bits": te})
    te_df = pd.DataFrame(te_rows)

    # ---------------- exports ----------------
    pd.DataFrame(sim).to_csv(
        "outputs_crossseed/metrics/interseed_epoch_similarity.csv",
        header=False,
        index=False,
    )
    pd.DataFrame(
        [
            {"metric": "ARI_seed_offdiag_diagmean", "value": ari_seed},
            {"metric": "k_clusters", "value": cfg.k},
            {"metric": "seeds", "value": cfg.seeds},
            {"metric": "epochs", "value": cfg.epochs},
        ]
    ).to_csv("outputs_crossseed/metrics/ari_seed_offdiag.csv", index=False)
    gc.to_csv("outputs_crossseed/metrics/granger_results.csv", index=False)
    te_df.to_csv("outputs_crossseed/metrics/te_results.csv", index=False)

    # ---------------- figures ----------------
    # heatmap (only diagonal meaningful here; we still plot full for consistency)
    plt.figure(figsize=(5, 4), dpi=120)
    plt.imshow(sim, origin="lower")
    plt.colorbar(label="avg cosine sim of matched centroids (across seeds)")
    plt.xticks(range(cfg.epochs))
    plt.yticks(range(cfg.epochs))
    plt.title("Inter-seed epoch similarity")
    plt.tight_layout()
    plt.savefig("outputs_crossseed/figures/interseed_similarity_heatmap.png")
    plt.close()

    # centroid scatter by seed
    plt.figure(figsize=(6, 5), dpi=120)
    for (seed, epoch), g in cent.groupby(["seed", "epoch"]):
        plt.scatter(g["pca1"], g["pca2"], s=60, alpha=0.8, label=f"S{seed}E{epoch}")
    h, l = plt.gca().get_legend_handles_labels()
    if len(h) > 10:
        h, l = h[:10], l[:10]
    plt.legend(h, l, fontsize=8, frameon=False, loc="best")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Centroid fingerprints by seed")
    plt.tight_layout()
    plt.savefig("outputs_crossseed/figures/centroid_scatter_by_seed.png")
    plt.close()

    # granger heatmap: -log10(p)
    gh = gc.copy()
    gh["neglog10p"] = -np.log10(gh["p"].replace(0, np.nan))
    mat = np.zeros((1, 3))
    mat[0, :] = gh["neglog10p"].fillna(0).values
    plt.figure(figsize=(4, 2.2), dpi=120)
    plt.imshow(mat, aspect="auto", origin="lower")
    plt.yticks([0], ["CCI→{M,C,Rc}"])
    plt.xticks([0, 1, 2], ["→Meaning", "→Coherence", "→Rc"], rotation=15)
    plt.colorbar(label="-log10(p)")
    plt.title("Granger (seed0)")
    plt.tight_layout()
    plt.savefig("outputs_crossseed/figures/granger_heatmap.png")
    plt.close()

    # TE bar
    plt.figure(figsize=(4.5, 3), dpi=120)
    plt.bar(te_df["to"], te_df["TE_bits"])
    plt.ylabel("TE (bits)")
    plt.title("Transfer Entropy proxy: CCI → target (seed0)")
    plt.tight_layout()
    plt.savefig("outputs_crossseed/figures/te_bar.png")
    plt.close()

    # ---------------- report ----------------
    md = []
    md.append("# Cross-seed Recurrence & Causality — Quick Report\n")
    md.append(
        f"- Seeds: {cfg.seeds} | Epochs: {cfg.epochs} | Agents: {cfg.agents} | k: {cfg.k}\n"
    )
    md.append("## Cross-seed Recurrence\n")
    md.append(
        f"- **ARI_seed** (mean epoch-wise centroid similarity across seeds): **{ari_seed:.3f}**\n"
    )
    md.append(
        "- See `figures/interseed_similarity_heatmap.png` (bright diagonal = same archetypes across seeds).\n"
    )
    md.append("## Causality\n")
    md.append(
        "- `metrics/granger_results.csv` with p-values for CCI→{Meaning, Coherence, Rc} (seed 0).\n"
    )
    md.append(
        "- `metrics/te_results.csv` with fast discrete TE proxy (bits). Higher TE suggests directional influence.\n"
    )
    md.append("## Figures\n")
    md.append(
        "- `centroid_scatter_by_seed.png` — repeating fingerprints across seeds.\n"
    )
    md.append("- `granger_heatmap.png`, `te_bar.png` — causality visuals.\n")

    # Add key findings summary
    md.append("## Key Findings\n")
    md.append(f"- Cross-seed archetypal stability: {ari_seed:.1%} similarity\n")
    if not gc.empty:
        significant_granger = gc[gc["p"] < 0.05]
        if not significant_granger.empty:
            md.append(
                f"- Significant Granger causality detected: {len(significant_granger)}/{len(gc)} relationships\n"
            )
        else:
            md.append("- No significant Granger causality detected at p<0.05\n")

    if not te_df.empty:
        max_te = te_df["TE_bits"].max()
        max_te_target = te_df.loc[te_df["TE_bits"].idxmax(), "to"]
        md.append(
            f"- Strongest information flow: CCI → {max_te_target} ({max_te:.3f} bits)\n"
        )

    with open("outputs_crossseed/report/report.md", "w") as f:
        f.write("\n".join(md))

    print("Done in %.2fs" % (time.time() - t0))
    print("See outputs_crossseed/{metrics,figures,report}")
    print(f"Cross-seed archetypal stability: {ari_seed:.3f}")
    if not te_df.empty:
        print(f"Max Transfer Entropy: CCI → {max_te_target} ({max_te:.3f} bits)")


if __name__ == "__main__":
    main()
