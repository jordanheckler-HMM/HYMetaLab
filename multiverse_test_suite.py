#!/usr/bin/env python3
"""
Multiverse Test Suite (≤ ~10 min total by default)

Covers six indirect "tests" or probes:
  A) CMB_SCARS          – look for collision-like anomalies (synthetic or via adapter)
  B) INFLATION_CHECK    – does an inflation-like prior best explain observed anisotropy?
  C) BH_SELECTION       – black-hole cosmology: do constants that favor BHs also favor persistence?
  D) FINE_TUNING_SWEEP  – survival across random "universe constants" (fine-tuning / selection)
  E) MANY_WORLDS_PROXY  – branch-heavy dynamics vs collapse-prone; which yields more persistent observers?
  F) MATH_INFO_ENSEMBLE – information-efficiency vs richness tradeoff; which "math universes" persist?

Adapters:
- If you have real data/modules (cmb_loader, inflation_model, bh_pop_model, etc.), edit the ADAPTERS section.
- Otherwise, fast synthetic fallbacks run so the pipeline completes in under ~10 minutes.

Outputs:
./discovery_results/multiverse_<stamp>/
  - Per-test: *_history.csv, *_summary.json, *_plots.png
  - REPORT.md : overview and pointers
"""

import datetime
import json
import math
import pathlib
import random
import time
import traceback
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =================== GLOBAL CONFIG ===================
SEED = 314159
TIME_BUDGET_MIN = 10  # hard cap
PER_CALL_TIMEOUT = 8

RUN_STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = pathlib.Path("./discovery_results") / f"multiverse_{RUN_STAMP}"
ROOT.mkdir(parents=True, exist_ok=True)

print("🌌 Starting Multiverse Test Suite...")
print(f"📁 Results will be saved to: {ROOT}")
print("🔬 Testing fundamental cosmological and physics questions...")

rng = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)


def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))


# =================== ADAPTERS (optional) ===================
def _safe_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


# Plug real modules here if you have them; else synthetic code runs
cmb_loader = _safe_import("cmb_loader")  # expect .load_map()->np.array or similar
inflation_model = _safe_import("inflation_model")  # expect .fit_and_compare(map)->dict
bh_pop_model = _safe_import(
    "bh_population"
)  # expect .estimate_bh_yield(constants)->float
universe_engine = _safe_import(
    "universe_engine"
)  # expect .simulate(constants)->metrics
quantum_brancher = _safe_import(
    "quantum_brancher"
)  # expect .run(branch_rate, decoh)->metrics
math_ensemble = _safe_import("math_ensemble")  # expect .sample_model(code)->metrics

print("🔧 Adapter status: Using synthetic fallbacks for fast testing")


# Utility: quick save figure safely
def savefig(path):
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# =================== A) CMB_SCARS ===================
def test_cmb_scars(outdir: pathlib.Path) -> dict[str, Any]:
    """
    Idea: If bubble-universe collisions occurred, they can leave circular/anisotropic "scars".
    We compute simple anomaly scores (rings/asymmetry) and compare to null.
    If you have real CMB data, plug it via cmb_loader.
    """
    print("  🌊 Testing A) CMB_SCARS - Looking for bubble collision signatures...")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    # get map (synthetic fallback: mixture of gaussian noise + injected ring anomalies)
    if cmb_loader and hasattr(cmb_loader, "load_map"):
        cmb = cmb_loader.load_map()  # shape (N,N)
    else:
        N = 128
        cmb = rng.normal(0, 1, (N, N))
        # randomly inject 0–2 ring anomalies
        n_rings = rng.integers(0, 3)
        print(f"     Injecting {n_rings} synthetic collision scars...")
        for _ in range(n_rings):
            cx, cy = rng.integers(16, N - 16, 2)
            r = rng.integers(8, 20)
            yy, xx = np.ogrid[:N, :N]
            mask = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - r) < 1.0
            cmb[mask] += rng.normal(3.0, 0.3)  # bright ring

    # Simple stats: ring score (radial power peak), hemispheric asymmetry
    N = cmb.shape[0]
    center = (N // 2, N // 2)
    yy, xx = np.indices(cmb.shape)
    r = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    r_bin = (r / 2).astype(int)
    df_prof = (
        pd.DataFrame({"rbin": r_bin.ravel(), "val": cmb.ravel()})
        .groupby("rbin")["val"]
        .std()
    )
    ring_score = float(np.nanmax(df_prof.values))
    hemi_asym = float(abs(cmb[:, : N // 2].mean() - cmb[:, N // 2 :].mean()))

    rows.append({"ring_score": ring_score, "hemi_asym": hemi_asym})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "CMB_SCARS_history.csv", index=False)

    # Null baseline via quick permutations
    print("     Computing statistical significance against null hypothesis...")
    null_rs, null_ha = [], []
    for _ in range(40):
        perm = np.random.permutation(cmb.ravel()).reshape(cmb.shape)
        dfp = (
            pd.DataFrame({"rbin": r_bin.ravel(), "val": perm.ravel()})
            .groupby("rbin")["val"]
            .std()
        )
        null_rs.append(np.nanmax(dfp.values))
        null_ha.append(abs(perm[:, : N // 2].mean() - perm[:, N // 2 :].mean()))
    p_ring = float((np.sum(np.array(null_rs) >= ring_score) + 1) / (len(null_rs) + 1))
    p_hemi = float((np.sum(np.array(null_ha) >= hemi_asym) + 1) / (len(null_ha) + 1))

    summary = {
        "ring_score": ring_score,
        "hemi_asym": hemi_asym,
        "p_ring": p_ring,
        "p_hemi": p_hemi,
    }
    with open(outdir / "CMB_SCARS_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"     → Ring anomaly score: {ring_score:.3f} (p={p_ring:.3f})")
    print(f"     → Hemispheric asymmetry: {hemi_asym:.3f} (p={p_hemi:.3f})")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cmb, aspect="auto", cmap="RdBu_r")
    plt.colorbar()
    plt.title("CMB Temperature Map")
    plt.subplot(1, 2, 2)
    plt.plot(df_prof.index, df_prof.values, "b-", label="Radial Profile STD")
    plt.axhline(y=ring_score, color="r", linestyle="--", label=f"Peak (p={p_ring:.3f})")
    plt.xlabel("Radius bin")
    plt.ylabel("Temperature STD")
    plt.legend()
    plt.title("Ring Anomaly Detection")
    savefig(outdir / "CMB_SCARS_analysis.png")

    return summary


# =================== B) INFLATION_CHECK ===================
def test_inflation_check(outdir: pathlib.Path) -> dict[str, Any]:
    """
    Idea: Compare simple inflation-like prior vs non-inflation prior on anisotropy stats.
    If you have a real model, plug inflation_model.fit_and_compare(map)
    """
    print("  🎈 Testing B) INFLATION_CHECK - Comparing cosmological models...")
    outdir.mkdir(parents=True, exist_ok=True)

    # synthetic "map stats"
    anisotropy = float(np.clip(rng.normal(1.0, 0.2), 0.3, 1.7))
    spectral_index = float(
        np.clip(rng.normal(0.965, 0.01), 0.9, 1.1)
    )  # Planck-like n_s near 0.965

    print(
        f"     Observed: spectral_index={spectral_index:.4f}, anisotropy={anisotropy:.3f}"
    )

    # simple scoring: if spectral_index near ~0.965 and anisotropy moderate → supports inflation-like
    infl_score = math.exp(-((spectral_index - 0.965) ** 2) / 0.0002) * math.exp(
        -((anisotropy - 1.0) ** 2) / 0.2
    )
    noninfl_score = math.exp(-((spectral_index - 1.02) ** 2) / 0.0002) * math.exp(
        -((anisotropy - 0.7) ** 2) / 0.2
    )
    bayes_factor = float(np.clip(infl_score / (noninfl_score + 1e-9), 0, 1e6))

    summary = {
        "anisotropy": anisotropy,
        "spectral_index": spectral_index,
        "bayes_factor_infl_vs_noninfl": bayes_factor,
    }
    with open(outdir / "INFLATION_CHECK_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"     → Inflation score: {infl_score:.4f}")
    print(f"     → Non-inflation score: {noninfl_score:.4f}")
    print(f"     → Bayes Factor (Inflation/Non): {bayes_factor:.2f}")

    # quick bar
    plt.figure()
    bars = plt.bar(["Inflation", "Non-Inflation"], [infl_score, noninfl_score])
    bars[0].set_color("green" if bayes_factor > 1 else "red")
    bars[1].set_color("red" if bayes_factor > 1 else "green")
    plt.ylabel("Model Score")
    plt.title(f"Inflation vs Non-inflation (BF≈{bayes_factor:.1f})")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "INFLATION_CHECK_comparison.png")
    return summary


# =================== C) BH_SELECTION ===================
def test_bh_selection(outdir: pathlib.Path) -> dict[str, Any]:
    """
    Idea: universes that maximize black-hole yield reproduce (Smolin-style).
    We sweep a few "constants" and proxy both BH yield and "persistence" (survival & CCI).
    """
    print("  🕳️  Testing C) BH_SELECTION - Black hole cosmological natural selection...")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    # parameter grid: (gravity G, cooling kappa, baryon_fraction b)
    grid = [
        (g, k, b)
        for g in np.linspace(0.8, 1.2, 5)
        for k in np.linspace(0.6, 1.4, 5)
        for b in np.linspace(0.1, 0.9, 5)
    ]
    rng.shuffle(grid)
    grid = grid[:64]  # keep it fast

    print(f"     Testing {len(grid)} universe parameter combinations...")

    for i, (G, kappa, bfrac) in enumerate(grid):
        if (i + 1) % 16 == 0:
            print(f"       Completed {i+1}/{len(grid)} universes...")

        if bh_pop_model and hasattr(bh_pop_model, "estimate_bh_yield"):
            bh_yield = float(
                bh_pop_model.estimate_bh_yield({"G": G, "kappa": kappa, "b": bfrac})
            )
        else:
            # synthetic: BH yield rises with gravity and cooling until fragmentation; too high collapses early
            base = (G * 1.2) * (kappa**0.5) * (bfrac**0.8)
            penalty = math.exp(-((G - 1.05) ** 2) / 0.02)  # favor slightly >1 gravity
            bh_yield = float(np.clip(base * penalty, 0, 10))

        # persistence proxy (higher if not too extreme)
        cci = float(
            np.clip(
                0.75 - 0.25 * abs(G - 1.0) - 0.15 * abs(kappa - 1.0) + 0.05 * bfrac,
                0,
                1,
            )
        )
        survival = float(
            np.clip(0.6 - 0.3 * abs(G - 1.0) - 0.2 * abs(kappa - 1.0), 0, 1)
        )
        rows.append(
            {
                "G": G,
                "kappa": kappa,
                "bfrac": bfrac,
                "bh_yield": bh_yield,
                "cci": cci,
                "survival": survival,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "BH_SELECTION_history.csv", index=False)

    # correlation between bh_yield and persistence
    corr_cci = float(np.corrcoef(df["bh_yield"], df["cci"])[0, 1])
    corr_surv = float(np.corrcoef(df["bh_yield"], df["survival"])[0, 1])
    best_row = df.sort_values("bh_yield", ascending=False).head(1).iloc[0]

    summary = {
        "corr_bh_vs_cci": corr_cci,
        "corr_bh_vs_survival": corr_surv,
        "best_params": {
            "G": float(best_row["G"]),
            "kappa": float(best_row["kappa"]),
            "bfrac": float(best_row["bfrac"]),
        },
        "best_bh_yield": float(best_row["bh_yield"]),
    }
    with open(outdir / "BH_SELECTION_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"     → Correlation BH yield vs CCI: {corr_cci:.3f}")
    print(f"     → Correlation BH yield vs Survival: {corr_surv:.3f}")
    print(
        f"     → Best universe: G={best_row['G']:.2f}, κ={best_row['kappa']:.2f}, b={best_row['bfrac']:.2f}"
    )

    # scatter plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(df["bh_yield"], df["cci"], alpha=0.6, c=df["G"], cmap="viridis")
    plt.colorbar(label="Gravity G")
    plt.xlabel("BH Yield")
    plt.ylabel("CCI")
    plt.title(f"BH yield vs CCI (r={corr_cci:.3f})")

    plt.subplot(1, 2, 2)
    plt.scatter(df["bh_yield"], df["survival"], alpha=0.6, c=df["kappa"], cmap="plasma")
    plt.colorbar(label="Cooling κ")
    plt.xlabel("BH Yield")
    plt.ylabel("Survival")
    plt.title(f"BH yield vs Survival (r={corr_surv:.3f})")
    savefig(outdir / "BH_SELECTION_analysis.png")
    return summary


# =================== D) FINE_TUNING_SWEEP ===================
def test_fine_tuning(outdir: pathlib.Path) -> dict[str, Any]:
    """
    Idea: random universes with random constants; what fraction survive with high CCI & low collapse?
    """
    print("  ⚖️  Testing D) FINE_TUNING_SWEEP - Anthropic principle investigation...")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    N = 500  # small but signals appear

    print(f"     Generating {N} random universes with varied fundamental constants...")

    for i in range(N):
        if (i + 1) % 100 == 0:
            print(f"       Simulated {i+1}/{N} universes...")

        constants = {
            "G": float(rng.normal(1.0, 0.25)),
            "alpha": float(rng.normal(1.0, 0.25)),  # fine-structure-like
            "lambda": float(rng.normal(1.0, 0.25)),  # cosmological-like
            "eta": float(rng.normal(1.0, 0.25)),  # matter/energy ratio
        }
        if universe_engine and hasattr(universe_engine, "simulate"):
            m = universe_engine.simulate(
                constants
            )  # expect dict with cci, survival, collapse
            cci = float(m.get("cci", 0.0))
            survival = float(m.get("survival", 0.0))
            collapse = float(m.get("collapse", 1.0))
        else:
            # synthetic persistence: near-1.0 constants do best; extremes fail
            dev = sum(abs(constants[k] - 1.0) for k in constants) / len(constants)
            cci = float(np.clip(0.85 - 0.6 * dev + 0.05 * rng.standard_normal(), 0, 1))
            survival = float(
                np.clip(0.65 - 0.7 * dev + 0.05 * rng.standard_normal(), 0, 1)
            )
            collapse = float(
                np.clip(0.35 + 1.2 * dev + 0.05 * rng.standard_normal(), 0, 1)
            )
        rows.append(
            {**constants, "cci": cci, "survival": survival, "collapse": collapse}
        )
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "FINE_TUNING_SWEEP_history.csv", index=False)

    survivors = df[
        (df["cci"] >= 0.75) & (df["survival"] >= 0.55) & (df["collapse"] <= 0.30)
    ]
    frac_survive = float(len(survivors) / len(df))

    summary = {
        "N": N,
        "survivor_fraction": frac_survive,
        "n_survivors": len(survivors),
        "median_cci": float(df["cci"].median()),
        "median_survival": float(df["survival"].median()),
        "median_collapse": float(df["collapse"].median()),
    }
    with open(outdir / "FINE_TUNING_SWEEP_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"     → Total universes: {N}")
    print(f"     → Survivors (high CCI, survival, low collapse): {len(survivors)}")
    print(f"     → Survival fraction: {frac_survive:.3f} ({frac_survive*100:.1f}%)")
    print(f"     → Median CCI: {df['cci'].median():.3f}")

    # triangle scatter
    plt.figure(figsize=(10, 6))
    plt.scatter(
        df["cci"],
        1.0 - df["collapse"],
        s=10,
        alpha=0.4,
        label=f"All ({N})",
        color="lightblue",
    )
    if len(survivors) > 0:
        s = survivors
        plt.scatter(
            s["cci"],
            1.0 - s["collapse"],
            s=18,
            alpha=0.8,
            label=f"Survivors ({len(s)})",
            color="red",
        )
    plt.xlabel("CCI")
    plt.ylabel("1 - Collapse Risk")
    plt.title(f"Fine-Tuning: Survivor Fraction = {frac_survive:.3f}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "FINE_TUNING_SWEEP_scatter.png")
    return summary


# =================== E) MANY_WORLDS_PROXY ===================
def test_many_worlds(outdir: pathlib.Path) -> dict[str, Any]:
    """
    Idea: compare branch-heavy universes vs collapse-heavy (fast decoherence).
    Measure expected number of persistent observers after T cycles.
    """
    print(
        "  🌀 Testing E) MANY_WORLDS_PROXY - Quantum branching vs collapse dynamics..."
    )
    outdir.mkdir(parents=True, exist_ok=True)
    T = 12
    reps = 100
    rows = []

    print(f"     Running {reps} simulations each for branching vs collapse regimes...")

    for mode in ["BRANCH_HEAVY", "COLLAPSE_HEAVY"]:
        print(f"       Testing {mode} regime...")
        for rep in range(reps):
            if quantum_brancher and hasattr(quantum_brancher, "run"):
                m = quantum_brancher.run(
                    branch_rate=0.6 if mode == "BRANCH_HEAVY" else 0.2,
                    decoherence=0.2 if mode == "BRANCH_HEAVY" else 0.6,
                    steps=T,
                )
                obs = float(m.get("observer_count", 0.0))
            else:
                # synthetic branching process
                branch_rate = 0.6 if mode == "BRANCH_HEAVY" else 0.2
                decoh = 0.2 if mode == "BRANCH_HEAVY" else 0.6
                obs = 1.0
                for t in range(T):
                    obs = obs * (1 + branch_rate - decoh) + rng.normal(0, 0.02)
                    obs = max(0.0, obs)
            rows.append({"mode": mode, "rep": rep, "observers": obs})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "MANY_WORLDS_history.csv", index=False)

    means = df.groupby("mode")["observers"].mean().to_dict()
    stds = df.groupby("mode")["observers"].std().to_dict()

    summary = {
        "mean_observers_branch_heavy": float(means.get("BRANCH_HEAVY", np.nan)),
        "mean_observers_collapse_heavy": float(means.get("COLLAPSE_HEAVY", np.nan)),
        "std_observers_branch_heavy": float(stds.get("BRANCH_HEAVY", np.nan)),
        "std_observers_collapse_heavy": float(stds.get("COLLAPSE_HEAVY", np.nan)),
    }
    with open(outdir / "MANY_WORLDS_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    branch_mean = means.get("BRANCH_HEAVY", 0)
    collapse_mean = means.get("COLLAPSE_HEAVY", 0)

    print(
        f"     → Branch-heavy regime: {branch_mean:.3f} ± {stds.get('BRANCH_HEAVY', 0):.3f} observers"
    )
    print(
        f"     → Collapse-heavy regime: {collapse_mean:.3f} ± {stds.get('COLLAPSE_HEAVY', 0):.3f} observers"
    )
    print(
        f"     → Ratio (Branch/Collapse): {branch_mean/max(collapse_mean, 0.001):.2f}"
    )

    plt.figure()
    modes = ["Branch-heavy", "Collapse-heavy"]
    values = [branch_mean, collapse_mean]
    errors = [stds.get("BRANCH_HEAVY", 0), stds.get("COLLAPSE_HEAVY", 0)]

    bars = plt.bar(modes, values, yerr=errors, capsize=5)
    bars[0].set_color("blue")
    bars[1].set_color("red")
    plt.ylabel("Expected Observer Count")
    plt.title("Many-Worlds: Branch vs Collapse Regimes")
    plt.grid(True, alpha=0.3)
    savefig(outdir / "MANY_WORLDS_comparison.png")
    return summary


# =================== F) MATH_INFO_ENSEMBLE ===================
def test_math_info(outdir: pathlib.Path) -> dict[str, Any]:
    """
    Idea: universes as mathematical/information objects.
    Test which balance of description-length (L) and generative richness (R) persists.
    Hypothesis: mid-L / high-R (compressed but expressive) persists best.
    """
    print("  📊 Testing F) MATH_INFO_ENSEMBLE - Mathematical universe selection...")
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    N = 400

    print(
        f"     Sampling {N} mathematical universes across complexity/richness space..."
    )

    for i in range(N):
        L = float(np.clip(rng.gamma(2.0, 0.6), 0.2, 6.0))  # description length
        R = float(np.clip(rng.normal(0.7, 0.2), 0.0, 1.5))  # richness
        if math_ensemble and hasattr(math_ensemble, "sample_model"):
            m = math_ensemble.sample_model({"L": L, "R": R})
            persistence = float(m.get("persistence", 0.0))
            cci = float(m.get("cci", 0.0))
        else:
            # synthetic: persistence rises with richness, falls if L too high or too low
            mid_penalty = math.exp(-((L - 2.0) ** 2) / 1.6)
            persistence = float(
                np.clip(
                    0.2 + 0.6 * R * mid_penalty + 0.05 * rng.standard_normal(), 0, 1
                )
            )
            cci = float(
                np.clip(
                    0.6 + 0.3 * R - 0.1 * abs(L - 2.0) + 0.05 * rng.standard_normal(),
                    0,
                    1,
                )
            )
        rows.append({"L": L, "R": R, "persistence": persistence, "cci": cci})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "MATH_INFO_ENSEMBLE_history.csv", index=False)

    # Where is the optimum?
    max_row = df.sort_values("persistence", ascending=False).iloc[0]
    top10 = df.nlargest(10, "persistence")

    summary = {
        "best_L": float(max_row["L"]),
        "best_R": float(max_row["R"]),
        "best_persistence": float(max_row["persistence"]),
        "best_cci": float(max_row["cci"]),
        "optimal_L_range": [float(top10["L"].min()), float(top10["L"].max())],
        "optimal_R_range": [float(top10["R"].min()), float(top10["R"].max())],
    }
    with open(outdir / "MATH_INFO_ENSEMBLE_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"     → Optimal complexity (L): {max_row['L']:.2f}")
    print(f"     → Optimal richness (R): {max_row['R']:.2f}")
    print(f"     → Best persistence: {max_row['persistence']:.3f}")
    print(f"     → Top 10 L range: [{top10['L'].min():.2f}, {top10['L'].max():.2f}]")
    print(f"     → Top 10 R range: [{top10['R'].min():.2f}, {top10['R'].max():.2f}]")

    # Create heatmap
    plt.figure(figsize=(10, 6))
    plt.scatter(df["L"], df["R"], c=df["persistence"], s=20, alpha=0.6, cmap="viridis")
    plt.colorbar(label="Persistence")
    plt.scatter(
        max_row["L"], max_row["R"], s=100, color="red", marker="*", label="Optimum"
    )
    plt.xlabel("Description Length (L)")
    plt.ylabel("Richness (R)")
    plt.title("Mathematical Universe Persistence Landscape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(outdir / "MATH_INFO_ENSEMBLE_landscape.png")
    return summary


# =================== RUNNER ===================
TESTS = {
    "CMB_SCARS": test_cmb_scars,
    "INFLATION_CHECK": test_inflation_check,
    "BH_SELECTION": test_bh_selection,
    "FINE_TUNING_SWEEP": test_fine_tuning,
    "MANY_WORLDS_PROXY": test_many_worlds,
    "MATH_INFO_ENSEMBLE": test_math_info,
}


def main(selected: list[str] = None):
    t0 = time.time()
    if not selected:
        # default: run all six quickly
        selected = list(TESTS.keys())

    print(f"\n🔬 Running {len(selected)} multiverse tests: {', '.join(selected)}")
    print(f"⏱️  Time budget: {TIME_BUDGET_MIN} minutes")

    overview = {}
    for i, name in enumerate(selected):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i+1}/{len(selected)}: {name}")
        print(f"{'='*60}")

        outd = ROOT / name
        try:
            res = TESTS[name](outd)
            overview[name] = res
            print(f"✅ {name} completed successfully")
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            overview[name] = {"error": str(e)}
            traceback.print_exc()

        elapsed = (time.time() - t0) / 60
        print(f"⏱️  Elapsed: {elapsed:.1f}/{TIME_BUDGET_MIN} minutes")

        if (time.time() - t0) > (TIME_BUDGET_MIN * 60):
            print(f"⏰ Time budget exceeded, stopping at test {i+1}")
            break

    with open(ROOT / "OVERVIEW.json", "w") as f:
        json.dump(overview, f, indent=2)

    # Simple report
    print("\n📝 Generating summary report...")
    with open(ROOT / "REPORT.md", "w") as f:
        f.write(f"# Multiverse Test Suite — {RUN_STAMP}\n\n")
        f.write("## Tests Completed\n\n")
        f.write("Tests run: " + ", ".join(overview.keys()) + "\n\n")

        f.write("## Test Descriptions\n\n")
        f.write(
            "- **CMB_SCARS**: Ring & hemispheric anomaly detection in cosmic microwave background\n"
        )
        f.write(
            "- **INFLATION_CHECK**: Bayesian comparison of inflation vs non-inflation models\n"
        )
        f.write(
            "- **BH_SELECTION**: Black hole cosmological natural selection correlations\n"
        )
        f.write(
            "- **FINE_TUNING_SWEEP**: Anthropic survivor fraction under random fundamental constants\n"
        )
        f.write(
            "- **MANY_WORLDS_PROXY**: Expected observers under quantum branching vs collapse dynamics\n"
        )
        f.write(
            "- **MATH_INFO_ENSEMBLE**: Persistence optimization in mathematical complexity/richness space\n\n"
        )

        f.write("## Key Findings Summary\n\n")
        for name, result in overview.items():
            if "error" not in result:
                f.write(f"### {name}\n")
                if name == "CMB_SCARS":
                    f.write(f"- Ring anomaly p-value: {result.get('p_ring', 'N/A')}\n")
                    f.write(
                        f"- Hemispheric asymmetry p-value: {result.get('p_hemi', 'N/A')}\n"
                    )
                elif name == "INFLATION_CHECK":
                    f.write(
                        f"- Bayes factor (Inflation/Non-inflation): {result.get('bayes_factor_infl_vs_noninfl', 'N/A')}\n"
                    )
                elif name == "BH_SELECTION":
                    f.write(
                        f"- BH yield vs CCI correlation: {result.get('corr_bh_vs_cci', 'N/A')}\n"
                    )
                    f.write(
                        f"- BH yield vs Survival correlation: {result.get('corr_bh_vs_survival', 'N/A')}\n"
                    )
                elif name == "FINE_TUNING_SWEEP":
                    f.write(
                        f"- Survivor fraction: {result.get('survivor_fraction', 'N/A')}\n"
                    )
                elif name == "MANY_WORLDS_PROXY":
                    branch = result.get("mean_observers_branch_heavy", "N/A")
                    collapse = result.get("mean_observers_collapse_heavy", "N/A")
                    f.write(f"- Branch-heavy observers: {branch}\n")
                    f.write(f"- Collapse-heavy observers: {collapse}\n")
                elif name == "MATH_INFO_ENSEMBLE":
                    f.write(
                        f"- Optimal complexity (L): {result.get('best_L', 'N/A')}\n"
                    )
                    f.write(f"- Optimal richness (R): {result.get('best_R', 'N/A')}\n")
                f.write("\n")

        f.write("\n## Files Generated\n\n")
        f.write("Each test produces:\n")
        f.write("- `*_history.csv`: Raw simulation data\n")
        f.write("- `*_summary.json`: Key results and parameters\n")
        f.write("- `*_*.png`: Visualization plots\n")

    print("\n✅ MULTIVERSE TEST SUITE COMPLETE!")
    print(f"📁 Results saved to: {ROOT}")
    print(
        f"🏆 {len([r for r in overview.values() if 'error' not in r])} of {len(selected)} tests completed successfully"
    )

    # Copy main report to easy access location
    import shutil

    print("\n📁 Copying key files to main directory...")
    shutil.copy(ROOT / "REPORT.md", "./MULTIVERSE_RESULTS.md")
    shutil.copy(ROOT / "OVERVIEW.json", "./MULTIVERSE_SUMMARY.json")

    print("✅ Easy access files created:")
    print("   - MULTIVERSE_RESULTS.md")
    print("   - MULTIVERSE_SUMMARY.json")


if __name__ == "__main__":
    import sys

    # Allow: python multiverse_test_suite.py CMB_SCARS INFLATION_CHECK ...
    sel = sys.argv[1:] if len(sys.argv) > 1 else None
    main(sel)
