#!/usr/bin/env python3
"""Love–Consciousness Interaction (LCI)

Loads Love Spectrum summaries and maps them to CCI modifiers, computing ΔCCI per love type and mixes.
Writes results to /mnt/data/love_cci_results.csv and a timestamped discovery_results folder.
"""
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


# ---------- Baseline & hyperparameters ----------
@dataclass
class CCIBaseline:
    calibration: float = 0.82
    coherence: float = 0.85
    emergence: float = 0.80
    noise: float = 0.18


BASE = CCIBaseline()

HP = {
    "k_coh": 0.50,
    "k_cal_from_open": 0.35,
    "k_em_from_open": 0.65,
    "k_noise": 0.60,
    "entropy_ref": 0.60,
    "softcap": 0.85,
}

LOVE_MIXES = {
    "Agape+Eros (70/30)": {"Agape": 0.7, "Eros": 0.3},
    "Agape+Philia (60/40)": {"Agape": 0.6, "Philia": 0.4},
    "Agape+Pragma (50/50)": {"Agape": 0.5, "Pragma": 0.5},
    "Agape+Ludus (60/40)": {"Agape": 0.6, "Ludus": 0.4},
}


# ---------- I/O helpers ----------
def find_latest_love_summary() -> Path | None:
    # Prefer /mnt/data JSON, then discovery_results/*/love_spectrum_summary.json, then CSV versions
    p_json = Path("/mnt/data/love_spectrum_summary.json")
    p_csv = Path("/mnt/data/love_spectrum_summary.csv")
    if p_json.exists():
        return p_json
    if p_csv.exists():
        return p_csv

    # search discovery_results
    dr = Path("discovery_results")
    if dr.exists():
        candidates = sorted(dr.glob("love_spectrum_*"))
        for c in reversed(candidates):
            j = c / "love_spectrum_summary.json"
            if j.exists():
                return j
            c_csv = c / "love_spectrum_summary.csv"
            if c_csv.exists():
                return c_csv
    return None


def load_love_summary() -> dict[str, dict[str, float]]:
    p = find_latest_love_summary()
    if p is None:
        # fallback defaults
        return {
            "Agape": {
                "mean_entropy_reduction": 0.61,
                "mean_coherence_boost": 0.70,
                "mean_openness_gain": 0.09,
            },
            "Eros": {
                "mean_entropy_reduction": 0.88,
                "mean_coherence_boost": 0.18,
                "mean_openness_gain": -0.70,
            },
            "Philia": {
                "mean_entropy_reduction": 0.76,
                "mean_coherence_boost": 0.39,
                "mean_openness_gain": -0.37,
            },
            "Pragma": {
                "mean_entropy_reduction": 0.76,
                "mean_coherence_boost": 0.38,
                "mean_openness_gain": -0.38,
            },
            "Storge": {
                "mean_entropy_reduction": 0.79,
                "mean_coherence_boost": 0.33,
                "mean_openness_gain": -0.45,
            },
            "Ludus": {
                "mean_entropy_reduction": 0.84,
                "mean_coherence_boost": 0.24,
                "mean_openness_gain": -0.60,
            },
        }

    if p.suffix.lower() == ".json":
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    else:
        out = {}
        with open(p, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                lt = row.get("love_type") or row.get("love")
                out[lt] = {
                    "mean_entropy_reduction": float(row["mean_entropy_reduction"]),
                    "mean_coherence_boost": float(row["mean_coherence_boost"]),
                    "mean_openness_gain": float(row["mean_openness_gain"]),
                }
        return out


# ---------- Modifier functions ----------
def softcap(x: float, cap: float) -> float:
    # Smooth cap to avoid runaway multipliers
    # simple sigmoid-like asymptote
    if x <= 0:
        return x
    return 1 - (1 - min(x, 10)) / (1 + min(x, 10) / ((1.0 / cap) - 1.0))


def modifiers_from_metrics(m: dict[str, float]) -> dict[str, float]:
    ent = m["mean_entropy_reduction"]
    coh = m["mean_coherence_boost"]
    opn = m["mean_openness_gain"]

    coh_mult = 1.0 + HP["k_coh"] * coh
    coh_mult = softcap(coh_mult, HP["softcap"])

    cal_mult = 1.0 + HP["k_cal_from_open"] * opn
    em_mult = 1.0 + HP["k_em_from_open"] * opn
    cal_mult = max(0.50, softcap(cal_mult, HP["softcap"]))
    em_mult = max(0.50, softcap(em_mult, HP["softcap"]))

    ent_excess = max(0.0, ent - HP["entropy_ref"])
    noise_mult = 1.0 - HP["k_noise"] * ent_excess
    noise_mult = max(0.05, noise_mult)

    return dict(cal=cal_mult, coh=coh_mult, em=em_mult, noise=noise_mult)


# ---------- CCI computation ----------
def cci(ca, co, em, nz) -> float:
    return (ca * co * em) / max(0.01, nz)


BASELINE_CCI = cci(BASE.calibration, BASE.coherence, BASE.emergence, BASE.noise)


def apply_modifiers_to_base(mods: dict[str, float]) -> float:
    ca = BASE.calibration * mods["cal"]
    co = BASE.coherence * mods["coh"]
    em = BASE.emergence * mods["em"]
    nz = BASE.noise * mods["noise"]
    return cci(ca, co, em, nz)


def mix_metrics(
    weights: dict[str, float], LOVE: dict[str, dict[str, float]]
) -> dict[str, float]:
    ent = sum(weights[k] * LOVE[k]["mean_entropy_reduction"] for k in weights)
    coh = sum(weights[k] * LOVE[k]["mean_coherence_boost"] for k in weights)
    opn = sum(weights[k] * LOVE[k]["mean_openness_gain"] for k in weights)
    return {
        "mean_entropy_reduction": ent,
        "mean_coherence_boost": coh,
        "mean_openness_gain": opn,
    }


# ---------- Main ----------
def run_and_export():
    LOVE = load_love_summary()
    rows = []

    def record(name: str, metrics: dict[str, float]):
        mods = modifiers_from_metrics(metrics)
        cci_new = apply_modifiers_to_base(mods)
        delta = cci_new - BASELINE_CCI
        pct = (delta / BASELINE_CCI) * 100.0
        rows.append(
            {
                "profile": name,
                "cal_mult": mods["cal"],
                "coh_mult": mods["coh"],
                "em_mult": mods["em"],
                "noise_mult": mods["noise"],
                "CCI_baseline": BASELINE_CCI,
                "CCI_new": cci_new,
                "Delta_CCI": delta,
                "Delta_CCI_%": pct,
                "mean_entropy_reduction": metrics["mean_entropy_reduction"],
                "mean_coherence_boost": metrics["mean_coherence_boost"],
                "mean_openness_gain": metrics["mean_openness_gain"],
            }
        )

    for lt, m in LOVE.items():
        record(lt, m)

    for mix_name, mix_w in LOVE_MIXES.items():
        metrics = mix_metrics(mix_w, LOVE)
        record(mix_name, metrics)

    # Prefer writing to /mnt/data if available (for shared mounts). If not writable, fall back to ./data/
    preferred = Path("/mnt")
    if preferred.exists() and os.access(preferred, os.W_OK):
        out_mnt = preferred / "data"
    else:
        out_mnt = Path("data")
    out_mnt.mkdir(parents=True, exist_ok=True)
    out_path = out_mnt / "love_cci_results.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # discovery_results copy
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dr = Path("discovery_results") / f"love_cci_{stamp}"
    dr.mkdir(parents=True, exist_ok=True)
    dr_path = dr / "love_cci_results.csv"
    with open(dr_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Print nice table
    def fmt(x):
        return f"{x:>8.3f}"

    print("LOVE → ΔCCI RESULTS")
    print(
        f"{'Profile':<28}  {'ΔCCI%':>8}  {'Cal×':>6} {'Coh×':>6} {'Em×':>6} {'Noise×':>7}  {'Open':>6} {'Coh↑':>6} {'Ent↓':>6}"
    )
    for r in rows:
        print(
            f"{r['profile']:<28}  {fmt(r['Delta_CCI_%'])}%  {fmt(r['cal_mult'])} {fmt(r['coh_mult'])} {fmt(r['em_mult'])} {fmt(r['noise_mult'])}  {fmt(r['mean_openness_gain'])} {fmt(r['mean_coherence_boost'])} {fmt(r['mean_entropy_reduction'])}"
        )

    print(
        f"\nExported: {out_path} (preferred mount: {'/mnt' if preferred.exists() else 'local'})"
    )
    print(f"Discovery copy: {dr_path}")
    return {"mnt": str(out_path), "discovery": str(dr_path)}


if __name__ == "__main__":
    run_and_export()
