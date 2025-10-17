#!/usr/bin/env python3
"""
debug_multi_run.py
Quick debug script to understand why only the first run produces valid ARI scores.
"""

import math
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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
    return float(np.mean([p[2] for p in pairs])) if pairs else 0.0


@dataclass
class Cfg:
    agents: int = 200
    epochs: int = 5
    seeds: int = 3
    steps: int = 800
    k: int = 4
    noise: float = 0.08


def synth_run(cfg: Cfg, seed: int):
    base_rng = np.random.RandomState(seed % 1000)
    local_rng = np.random.RandomState(seed)
    soft_seed(seed)

    A, E, T = cfg.agents, cfg.epochs, cfg.steps

    base_C = np.stack(
        [
            [1.0, 0.8, 0.7, 0.25, 0.35],
            [0.6, 0.9, 0.6, 0.4, 0.55],
            [0.35, 0.55, 0.9, 0.6, 0.6],
            [0.85, 0.6, 0.5, 0.3, 0.45],
        ]
    ) + base_rng.normal(0, 0.02, (cfg.k, 5))

    ep_tf = []
    for e in range(E):
        if e in (0, 2):
            R = np.eye(5) + local_rng.normal(0, 0.015, (5, 5))
            b = np.array([0.02, 0, 0.01, -0.01, 0.0]) + local_rng.normal(0, 0.005, 5)
        elif e in (1, 3):
            R = np.eye(5) + local_rng.normal(0, 0.015, (5, 5))
            b = np.array([0, -0.02, 0.02, 0, 0.01]) + local_rng.normal(0, 0.005, 5)
        else:
            R = np.eye(5) + local_rng.normal(0, 0.025, (5, 5))
            b = np.array([0.03, -0.01, 0.02, 0.01, 0]) + local_rng.normal(0, 0.008, 5)
        ep_tf.append((R, b))

    rows = []
    labels = np.tile(np.arange(cfg.k), math.ceil(A / cfg.k))[:A]
    local_rng.shuffle(labels)

    t = np.linspace(0, 2 * np.pi, T)
    wiggle = np.stack(
        [
            0.025 * np.cos(t),
            0.025 * np.sin(t),
            0.018 * np.cos(2 * t),
            -0.025 * np.cos(t),
            0.012 * np.sin(1.5 * t),
        ],
        axis=1,
    )

    for e in range(E):
        R, b = ep_tf[e]
        epoch_C = (base_C @ R.T) + b
        agent_offsets = local_rng.normal(0, 0.035, (A, 5))
        for a in range(A):
            c = epoch_C[labels[a]]
            noise = local_rng.normal(0, cfg.noise * 0.8, (T, 5))
            series = c + agent_offsets[a] + wiggle + noise
            series[:, 3] = np.clip(series[:, 3], 0, 1)
            series[:, 4] = np.clip(series[:, 4], 0, 1)
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


def debug_run(cfg: Cfg, seed_offset: int = 0):
    print(f"\n=== DEBUG RUN (seed_offset={seed_offset}) ===")

    # Generate data
    dfs = [synth_run(cfg, s + seed_offset) for s in range(cfg.seeds)]
    df = pd.concat(dfs, ignore_index=True)
    feats = ["CCI", "Meaning", "Coherence", "Rc", "epsilon"]

    print(f"Generated {len(df)} data points")
    print(f"Feature ranges: {df[feats].describe().loc[['min','max']].round(3)}")

    # Scale and cluster
    scaler = StandardScaler().fit(df[feats].values)
    cents = []
    E, S, K = cfg.epochs, cfg.seeds, cfg.k

    for (seed, epoch), g in df.groupby(["seed", "epoch"]):
        Xg = scaler.transform(g[feats].values)
        print(f"  Seed {seed}, Epoch {epoch}: {len(Xg)} points")

        while len(Xg) < cfg.k:
            if len(Xg) > 0:
                noise_pt = Xg[-1] + np.random.normal(0, 0.01, Xg.shape[1])
                Xg = np.vstack([Xg, noise_pt])
            else:
                Xg = np.random.normal(0, 0.1, (cfg.k, len(feats)))

        km = KMeans(n_clusters=cfg.k, n_init=10, random_state=seed * 101 + epoch)
        km.fit(Xg)
        C = scaler.inverse_transform(km.cluster_centers_)

        print(f"    Cluster centers shape: {C.shape}")
        for j in range(cfg.k):
            cents.append(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "cluster": j,
                    **{f: C[j, i] for i, f in enumerate(feats)},
                }
            )

    cent = pd.DataFrame(cents)
    print(f"Total centroids: {len(cent)}")

    # Calculate ARI
    M = {}
    for (s, e), g in cent.groupby(["seed", "epoch"]):
        C = g.sort_values("cluster")[feats].values
        while C.shape[0] < K:
            C = np.vstack([C, C[-1] + np.random.normal(0, 0.01, len(feats))])
        M[(s, e)] = C[:K]
        print(f"  M[({s},{e})] shape: {M[(s,e)].shape}")

    diag = []
    for e in range(E):
        sims = []
        for i in range(S):
            for j in range(i + 1, S):
                if (i, e) in M and (j, e) in M:
                    sim = greedy_match(M[(i, e)], M[(j, e)])
                    sims.append(sim)
                    print(f"    Epoch {e}, Seeds {i}-{j}: similarity = {sim:.4f}")
        epoch_avg = float(np.mean(sims)) if sims else 0.0
        diag.append(epoch_avg)
        print(f"  Epoch {e} average similarity: {epoch_avg:.4f}")

    ari_seed = np.mean(diag) if diag else 0.0
    print(f"Final ARI_seed: {ari_seed:.4f}")
    return ari_seed


if __name__ == "__main__":
    cfg = Cfg()

    # Test first few runs
    for offset in [0, 1000, 2000]:
        debug_run(cfg, offset)
