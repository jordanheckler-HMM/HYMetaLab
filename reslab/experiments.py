import argparse

import numpy as np
import pandas as pd

from .adapter import run_sim
from .metrics import align_index_proxy, bootstrap_ci, cci, survival_rate_from_any
from .plotting import heatmap, line_with_ci
from .utils import ensure_dir, merge_cfg, set_seed


def _replicate(cfg, params):
    srates, ccis = [], []
    for r in range(cfg["replicates"]):
        out = run_sim(
            n_agents=cfg["n_agents"],
            ticks=cfg["ticks"],
            shock_level=params.get("shock_level", cfg["defaults"]["shock_level"]),
            coordination_strength=params.get(
                "coordination_strength", cfg["defaults"]["coordination_strength"]
            ),
            goal_inequality=params.get(
                "goal_inequality", cfg["defaults"]["goal_inequality"]
            ),
            noise=params.get("noise", cfg["defaults"]["noise"]),
            memory_cap=params.get("memory_cap", cfg["defaults"]["memory_cap"]),
            energy_drift=params.get("energy_drift", cfg["defaults"]["energy_drift"]),
            seed=cfg["seed"] + r,
        )
        srates.append(survival_rate_from_any(out["survived"]))
        c = cci(
            out.get("calibration"),
            out.get("coherence"),
            out.get("emergence"),
            out.get("noise"),
        )
        if c is not None:
            ccis.append(c)
    return np.array(srates), (np.array(ccis) if ccis else None)


def param_sweep(cfg):
    sp = cfg["sweep_param"]
    values = cfg["values"]
    fixed = cfg.get("fixed", {})
    rows = []
    for v in values:
        params = fixed | {sp: v}
        s, c = _replicate(cfg, params)
        m_s, (lo_s, hi_s) = float(np.mean(s)), bootstrap_ci(s, **cfg["bootstrap"])
        row = {sp: v, "survival_mean": m_s, "survival_lo": lo_s, "survival_hi": hi_s}
        # alignment proxy from knobs (coord & inequality)
        coord = params.get(
            "coordination_strength", cfg["defaults"]["coordination_strength"]
        )
        ineq = params.get("goal_inequality", cfg["defaults"]["goal_inequality"])
        row["align_index_proxy"] = align_index_proxy(coord, ineq)
        rows.append(row)
    df = pd.DataFrame(rows)
    ensure_dir(cfg["out_csv"])
    df.to_csv(cfg["out_csv"], index=False)
    x = df[sp].values
    line_with_ci(
        x,
        df["survival_mean"].values,
        df["survival_lo"].values,
        df["survival_hi"].values,
        sp,
        "Survival",
        cfg.get("label", "param sweep"),
        cfg["out_fig"],
    )
    return df


def chaos(cfg):
    ch = cfg["chaos"]
    steps = int(ch["steps"])
    base = ch["base_shock"]
    p = ch["spike_prob"]
    spike_to = ch["spike_to"]
    smooth_to = ch["smooth_to"]
    fixed = cfg.get("fixed", {})
    rows = []
    for t in range(steps):
        shock = spike_to if (np.random.rand() < p) else base
        params = fixed | {"shock_level": shock}
        s, _ = _replicate(cfg, params)
        m_s = float(np.mean(s))
        rows.append({"t": t, "shock": shock, "survival_mean": m_s})
    import pandas as pd

    df = pd.DataFrame(rows)
    ensure_dir(cfg["out_csv"])
    df.to_csv(cfg["out_csv"], index=False)
    # plot
    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(df["t"], df["shock"])
    plt.ylabel("Shock")
    plt.title(cfg.get("label", "chaos"))
    plt.subplot(2, 1, 2)
    plt.plot(df["t"], df["survival_mean"])
    plt.ylabel("Survival")
    plt.xlabel("t")
    plt.tight_layout()
    plt.savefig(cfg["out_fig"])
    plt.close()
    return df


def grid(cfg):
    gx = cfg["grid"]["coordination_strength"]
    gy = cfg["grid"]["goal_inequality"]
    fixed = cfg.get("fixed", {})
    Z = np.zeros((len(gy), len(gx)))
    rows = []
    for j, ineq in enumerate(gy):
        for i, coord in enumerate(gx):
            params = fixed | {"coordination_strength": coord, "goal_inequality": ineq}
            s, _ = _replicate(cfg, params)
            m_s = float(np.mean(s))
            Z[j, i] = m_s
            rows.append(
                {
                    "coordination_strength": coord,
                    "goal_inequality": ineq,
                    "survival_mean": m_s,
                }
            )
    import pandas as pd

    df = pd.DataFrame(rows)
    ensure_dir(cfg["out_csv"])
    df.to_csv(cfg["out_csv"], index=False)
    heatmap(
        Z,
        [str(x) for x in gx],
        [str(y) for y in gy],
        "coordination_strength",
        "goal_inequality",
        cfg.get("label", "grid"),
        cfg["out_fig"],
    )
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = merge_cfg([args.config])
    set_seed(cfg["seed"])
    if cfg["experiment"] == "param_sweep":
        param_sweep(cfg)
    elif cfg["experiment"] == "chaos":
        chaos(cfg)
    elif cfg["experiment"] == "grid":
        grid(cfg)
    else:
        raise SystemExit("unknown experiment")


if __name__ == "__main__":
    main()
