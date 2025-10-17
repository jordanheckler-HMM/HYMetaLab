from __future__ import annotations

"""Phase31b adapter that uses the sim bridge to run real simulations.

This module provides two entrypoints used by the pipeline:
- run_agents(cfg)
- run_civ(cfg)

Each entrypoint runs the configured suite of small experiments (as
specified by the prereg YAML) using the real sim via adapters.sim_bridge.run_simulation,
writes per-run CSVs to discovery_results/phase31b_uclv/, and writes a
summary_{domain}.json containing simple linear-fit diagnostics.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

from adapters.sim_bridge import run_simulation

OUT = Path("discovery_results/phase31b_uclv")
OUT.mkdir(parents=True, exist_ok=True)


def _writetable(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with open(path, "w") as f:
            f.write("")
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fit(x: list[float], y: list[float]):
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = sum((xi - mx) ** 2 for xi in x)
    m = num / den if den > 0 else float("nan")
    b = my - (m * mx if m == m else 0)
    yhat = [m * xi + b for xi in x]
    ss_res = sum((yi - yh) ** 2 for yi, yh in zip(y, yhat))
    ss_tot = sum((yi - my) ** 2 for yi in y)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return m, b, r2


def _run_block(name: str, seeds: list[int], epochs: int, sweep: dict, controls: dict):
    rows: list[dict] = []
    if name == "TestA_vary_epsilon":
        eps_list = sweep.get("epsilon", [])
        cci_t = controls.get("cci_target")
        eta_t = controls.get("eta_target")
        for seed in seeds:
            for eps in eps_list:
                out = run_simulation(
                    epsilon=eps,
                    seed=seed,
                    epochs=epochs,
                    cci_target=cci_t,
                    eta_target=eta_t,
                )
                rows.append(
                    dict(
                        test=name,
                        seed=seed,
                        epsilon=eps,
                        cci=out.cci,
                        eta=out.eta,
                        resilience=out.resilience,
                        survival_rate=out.survival_rate,
                        hazard=out.hazard,
                    )
                )
    elif name == "TestB_vary_cci":
        cci_list = sweep.get("cci_target", [])
        eps = controls.get("epsilon")
        eta_t = controls.get("eta_target")
        for seed in seeds:
            for cci_t in cci_list:
                out = run_simulation(
                    epsilon=eps,
                    seed=seed,
                    epochs=epochs,
                    cci_target=cci_t,
                    eta_target=eta_t,
                )
                rows.append(
                    dict(
                        test=name,
                        seed=seed,
                        epsilon=eps,
                        cci=out.cci,
                        eta=out.eta,
                        resilience=out.resilience,
                        survival_rate=out.survival_rate,
                        hazard=out.hazard,
                    )
                )
    elif name == "TestC_vary_eta":
        eta_list = sweep.get("eta_target", [])
        eps = controls.get("epsilon")
        cci_t = controls.get("cci_target")
        for seed in seeds:
            for eta_t in eta_list:
                out = run_simulation(
                    epsilon=eps,
                    seed=seed,
                    epochs=epochs,
                    cci_target=cci_t,
                    eta_target=eta_t,
                )
                rows.append(
                    dict(
                        test=name,
                        seed=seed,
                        epsilon=eps,
                        cci=out.cci,
                        eta=out.eta,
                        resilience=out.resilience,
                        survival_rate=out.survival_rate,
                        hazard=out.hazard,
                    )
                )
    else:
        raise ValueError(f"Unknown test {name}")
    return rows


def _run_suite(domain: str, config: dict):
    seeds = config["seeds"]
    epochs = config["epochs"]
    suite = config["suite"]
    all_rows: list[dict] = []
    for test in suite:
        all_rows += _run_block(
            test["name"], seeds, epochs, test.get("sweep", {}), test.get("controls", {})
        )

    out_csv = OUT / f"runs_{domain}.csv"
    _writetable(all_rows, out_csv)

    import collections

    fits = []
    by_test = collections.defaultdict(list)
    for r in all_rows:
        by_test[r["test"]].append(r)
    import math

    for test, rr in by_test.items():
        if test == "TestA_vary_epsilon":
            x = [r["epsilon"] for r in rr]
            xname = "epsilon"
            y = [r["resilience"] for r in rr]
            # filter finite pairs
            pairs = [
                (xi, yi)
                for xi, yi in zip(x, y)
                if xi is not None
                and yi is not None
                and math.isfinite(float(xi))
                and math.isfinite(float(yi))
            ]
        elif test == "TestB_vary_cci":
            x = [r["cci"] for r in rr]
            xname = "cci"
            y = [r["resilience"] for r in rr]
            pairs = [
                (xi, yi)
                for xi, yi in zip(x, y)
                if xi is not None
                and yi is not None
                and math.isfinite(float(xi))
                and math.isfinite(float(yi))
            ]
        else:
            # inverse-eta; avoid division by zero or non-finite eta values
            x_raw = []
            y = [r["resilience"] for r in rr]
            for r in rr:
                eta_val = r.get("eta")
                try:
                    eta_f = float(eta_val)
                    if eta_f == 0 or not math.isfinite(eta_f):
                        x_raw.append(float("nan"))
                    else:
                        x_raw.append(1.0 / eta_f)
                except Exception:
                    x_raw.append(float("nan"))
            x = x_raw
            xname = "inv_eta"
            pairs = [
                (xi, yi)
                for xi, yi in zip(x, y)
                if xi is not None
                and yi is not None
                and math.isfinite(float(xi))
                and math.isfinite(float(yi))
            ]

        if not pairs:
            m = float("nan")
            b = float("nan")
            r2 = float("nan")
        else:
            xs, ys = zip(*pairs)
            m, b, r2 = _fit(list(xs), list(ys))
        fits.append(
            dict(domain=domain, test=test, x=xname, slope=m, intercept=b, r2=r2)
        )
    summary = {"timestamp": datetime.utcnow().isoformat() + "Z", "fits": fits}
    with open(OUT / f"summary_{domain}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[31b] Wrote {out_csv.name} and summary_{domain}.json")
    return summary


def run_agents(cfg: dict):
    return _run_suite("agents", cfg)


def run_civ(cfg: dict):
    return _run_suite("civ", cfg)
