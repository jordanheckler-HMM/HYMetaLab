#!/usr/bin/env python3
"""
Truth Stability — Phase 3.1 (Parameter Drift Visualization)
Replays Phase-3 agent training to log weights per cycle and plot Δw.
"""
import argparse
import json
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logloss_grad(w, X, y):
    p = sigmoid(X.dot(w))
    grad = X.T.dot(p - y) / X.shape[0]
    return grad, p


def design_matrix(A, B, C, wide=False):
    ones = np.ones_like(A, dtype=float)
    if not wide:
        X = np.vstack([ones, A.astype(float), B.astype(float), C.astype(float)]).T
    else:
        AB = (A * B).astype(float)
        BC = (B * C).astype(float)
        X = np.vstack(
            [ones, A.astype(float), B.astype(float), C.astype(float), AB, BC]
        ).T
    return X


def simulate_cycle_W1(n, pA, pB, pC, rng):
    A = (rng.random(n) < pA).astype(int)
    B = (rng.random(n) < pB).astype(int)
    C = (rng.random(n) < pC).astype(int)
    eps = rng.normal(0.0, 0.08, size=n)
    S = 0.55 * A + 0.30 * B - 0.35 * C + eps
    S = np.clip(S, 0.0, 1.0)
    Y = (S > 0.5).astype(int)
    return A, B, C, S, Y


def simulate_cycle_W2(n, pA, pB, pC, rng):
    A = (rng.random(n) < pA).astype(int)
    B = (rng.random(n) < pB).astype(int)
    C = (rng.random(n) < pC).astype(int)
    eps = rng.normal(0.0, 0.08, size=n)
    S = 0.15 * A + 0.50 * B - 0.10 * C + 0.20 * (A * B) - 0.25 * (B * C) + eps
    S = np.clip(S, 0.0, 1.0)
    Y = (S > 0.5).astype(int)
    return A, B, C, S, Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6000)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RANDOM_SEED = int(args.seed)
    TRIALS = int(args.trials)
    CYCLES = int(args.cycles)
    per_cycle = max(1, TRIALS // CYCLES)

    rng = np.random.default_rng(RANDOM_SEED)

    # per-cycle probabilities schedule (same deterministic schedule)
    base = np.linspace(0.2, 0.8, CYCLES)
    cycle_ps = []
    for i in range(CYCLES):
        pA = float(np.clip(base[i] + 0.1 * np.sin(i * 1.3), 0.05, 0.95))
        pB = float(np.clip(base[-i] + 0.1 * np.cos(i * 0.9), 0.05, 0.95))
        pC = float(np.clip(0.5 + 0.3 * np.sin(i * 0.6 + 0.4), 0.05, 0.95))
        cycle_ps.append((pA, pB, pC))

    cycles_split = CYCLES // 2

    # Agents init (mirror Phase3)
    agents = {
        "Simple": {"wide": False, "lr": 0.4, "w": None, "prev_delta": None},
        "Wide": {"wide": True, "lr": 0.3, "w": None, "prev_delta": None},
        "Sticky": {
            "wide": True,
            "lr": 0.2,
            "w": None,
            "prev_delta": None,
            "momentum": 0.8,
        },
    }
    for name, a in agents.items():
        d = 4 if not a["wide"] else 6
        a["w"] = np.zeros(d, dtype=float)
        a["prev_delta"] = np.zeros(d, dtype=float)

    # record weights per cycle
    weights_log = {name: [] for name in agents}

    # run cycles and record weights after each cycle
    for ci, (pA, pB, pC) in enumerate(cycle_ps):
        if ci < cycles_split:
            A, B, C, S, Y = simulate_cycle_W1(per_cycle, pA, pB, pC, rng)
        else:
            A, B, C, S, Y = simulate_cycle_W2(per_cycle, pA, pB, pC, rng)

        for name, a in agents.items():
            X = design_matrix(A, B, C, wide=a["wide"])
            w = a["w"]
            # one-step update like Phase3
            grad, _ = logloss_grad(w, X, Y)
            delta = -a["lr"] * grad
            if a.get("momentum") is not None:
                delta = (1 - a["momentum"]) * delta + a["momentum"] * a["prev_delta"]
            w = w + delta
            a["w"] = w
            a["prev_delta"] = delta
            # store a copy
            weights_log[name].append(w.copy())

    # compute deltas per agent (length C-1)
    deltas = {}
    for name, ws in weights_log.items():
        arr = np.vstack(ws)  # shape (C, d)
        diffs = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
        deltas[name] = diffs

    # compute metrics: mean_pre (cycles 1-5), mean_post (cycles 7-12), spike_ratio (delta at switch / mean_pre)
    rows = []
    for name, diffs in deltas.items():
        # diffs indexed 0..C-2 corresponding to changes between cycle t and t+1 for t=1..C-1
        pre_indices = list(range(0, cycles_split - 1))  # 0..4 for split=6
        post_indices = list(range(cycles_split, CYCLES - 1))  # 6..10 for split=6
        mean_pre = (
            float(np.mean(diffs[pre_indices])) if len(pre_indices) > 0 else float("nan")
        )
        mean_post = (
            float(np.mean(diffs[post_indices]))
            if len(post_indices) > 0
            else float("nan")
        )
        # spike at switch is between cycle 6 and 7 => index cycles_split-1 (which is 5)
        spike_idx = cycles_split - 1
        spike = float(diffs[spike_idx])
        spike_ratio = float(spike / mean_pre) if mean_pre > 0 else float("inf")
        max_idx = (
            int(np.argmax(diffs)) + 1
        )  # report cycle number t where max delta is between t and t+1
        rows.append(
            {
                "agent": name,
                "mean_pre": round(mean_pre, 4),
                "mean_post": round(mean_post, 4),
                "spike_ratio": round(spike_ratio, 3),
                "max_drift_cycle": int(max_idx),
            }
        )

    df_drift = pd.DataFrame(rows)

    # save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("discovery_results", f"truth_phase3_drift_{timestamp}")
    os.makedirs(outdir, exist_ok=True)
    df_drift.to_csv(os.path.join(outdir, "drift_summary.csv"), index=False)

    # plot param drift
    plt.figure(figsize=(10, 5))
    x = np.arange(1, CYCLES)
    for name in deltas:
        plt.plot(x, deltas[name], marker="o", label=name)
    # vertical dashed line at switch after cycle 'cycles_split'
    plt.axvline(x=cycles_split + 0.0, color="k", linestyle="--")
    plt.xlabel("Cycle (t) — Δw between t and t+1")
    plt.ylabel("||Δw||_2")
    plt.title("Parameter Drift per Cycle (Phase 3.1)")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    # add text box with spike ratios
    text_lines = []
    for r in rows:
        text_lines.append(
            f"{r['agent']}: spike={r['spike_ratio']} pre={r['mean_pre']} post={r['mean_post']}"
        )
    plt.gcf().text(0.02, 0.02, "\n".join(text_lines), fontsize=9, va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "param_drift.png"), dpi=150)
    plt.close()

    # README
    readme = []
    readme.append("# Truth Stability — Phase 3.1 (Parameter Drift)")
    readme.append(f"Timestamp: {timestamp}")
    readme.append(f"Seed: {RANDOM_SEED}")
    readme.append(f"Cycles: {CYCLES}")
    readme.append("")
    readme.append(df_drift.to_markdown(index=False))
    readme.append("")
    readme.append(
        "Notes: Δw measures how strongly agents updated their internal models; spike at switch expected."
    )
    with open(os.path.join(outdir, "README.md"), "w") as f:
        f.write("\n\n".join(readme))

    params = {
        "timestamp": timestamp,
        "seed": RANDOM_SEED,
        "cycles": CYCLES,
        "trials": TRIALS,
    }
    with open(os.path.join(outdir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)

    # console summary
    print("\nAgent   spike_ratio  mean_pre  mean_post")
    for r in rows:
        print(
            f"{r['agent']:<7} {r['spike_ratio']:<12} {r['mean_pre']:<8} {r['mean_post']}"
        )
    print(f"Plot saved: {os.path.abspath(os.path.join(outdir, 'param_drift.png'))}")


if __name__ == "__main__":
    main()
