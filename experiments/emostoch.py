#!/usr/bin/env python3
"""EMO-STOCH: Emotional Stoichiometry & Kinetics

Mass-action style kinetics for love/emotion species. Writes trajectories and summary CSVs.
Auto-detects seeds from /mnt or discovery_results; prefers /mnt for outputs but falls back to local paths.
"""
import csv
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ------------------------
# 0) Simulation parameters
# ------------------------
T = 200  # steps
dt = 0.05  # step size
tau = 1.0  # "emotional temperature" (higher = more reactivity)
leak = 0.01  # global decay to prevent divergence

# ------------------------
# 1) Species & state vector
# ------------------------
SPECIES = [
    "Agape",
    "Eros",
    "Philia",
    "Pragma",
    "Storge",
    "Ludus",
    "Joy",
    "Gratitude",
    "Curiosity",
    "Anger",
    "Fear",
    "Sadness",
    "Awe",
    "Compassion",
    "Hope",
    "Openness",
    "Coherence",
    "Entropy",
]

# Seed concentrations (0..1). We'll try to bootstrap from love_spectrum_summary.json if available.
x0 = {s: 0.05 for s in SPECIES}


def load_love_seeds():
    # Try /mnt first
    candidates = [
        Path("/mnt/data/love_spectrum_summary.json"),
        Path("/mnt/data/love_spectrum_summary.csv"),
    ]
    # then search discovery_results
    dr = Path("discovery_results")
    if dr.exists():
        for c in sorted(dr.glob("love_spectrum_*"), reverse=True):
            j = c / "love_spectrum_summary.json"
            if j.exists():
                candidates.append(j)
            csvf = c / "love_spectrum_summary.csv"
            if csvf.exists():
                candidates.append(csvf)

    for p in candidates:
        if p.exists():
            try:
                if p.suffix.lower() == ".json":
                    with open(p, encoding="utf-8") as f:
                        love = json.load(f)
                else:
                    love = {}
                    with open(p, newline="", encoding="utf-8") as f:
                        r = csv.DictReader(f)
                        for row in r:
                            lt = row.get("love_type") or row.get("love")
                            love[lt] = {
                                "mean_entropy_reduction": float(
                                    row["mean_entropy_reduction"]
                                ),
                                "mean_coherence_boost": float(
                                    row["mean_coherence_boost"]
                                ),
                                "mean_openness_gain": float(row["mean_openness_gain"]),
                            }
                # map to x0
                for name, m in love.items():
                    if name in x0:
                        x0[name] = max(
                            0.02,
                            min(
                                1.0,
                                0.5 * m.get("mean_coherence_boost", 0.3)
                                + 0.5 * (1.0 - m.get("mean_entropy_reduction", 0.4)),
                            ),
                        )
                ag = love.get("Agape")
                if ag:
                    x0["Openness"] = max(
                        0.05, min(1.0, 0.5 + 0.5 * ag.get("mean_openness_gain", 0.0))
                    )
                x0["Coherence"] = min(
                    1.0,
                    sum(
                        x0.get(s, 0.0)
                        for s in [
                            "Agape",
                            "Philia",
                            "Pragma",
                            "Compassion",
                            "Gratitude",
                        ]
                    )
                    / 3.0,
                )
                x0["Entropy"] = max(0.05, 0.5 - 0.25 * x0["Coherence"])
                print(f"Loaded love seeds from {p}")
                return
            except Exception as e:
                print(f"Failed to read {p}: {e}")
    # if none found, keep defaults


load_love_seeds()

# ---------------------------------
# 2) Stoichiometric reaction schema
# ---------------------------------
REACTIONS = [
    {
        "label": "Agape expansion",
        "eq": {
            "reactants": {"Agape": 1, "Openness": 1},
            "products": {"Openness": 2, "Coherence": 1},
        },
        "k": 0.8 * tau,
        "k_rev": 0.05,
    },
    {
        "label": "Compassion healing",
        "eq": {
            "reactants": {"Compassion": 1, "Entropy": 1},
            "products": {"Coherence": 1},
        },
        "k": 0.7 * tau,
        "k_rev": 0.02,
    },
    {
        "label": "Gratitude ordering",
        "eq": {
            "reactants": {"Gratitude": 1, "Entropy": 1},
            "products": {"Coherence": 1},
        },
        "k": 0.6 * tau,
        "k_rev": 0.02,
    },
    {
        "label": "Eros×Anger → Ludus+Curiosity",
        "eq": {
            "reactants": {"Eros": 1, "Anger": 1},
            "products": {"Ludus": 1, "Curiosity": 1},
        },
        "k": 0.9 * tau,
        "k_rev": 0.1,
    },
    {
        "label": "Philia stabilization",
        "eq": {"reactants": {"Philia": 1}, "products": {"Coherence": 1}},
        "k": 0.4 * tau,
        "k_rev": 0.05,
    },
    {
        "label": "Pragma structure",
        "eq": {"reactants": {"Pragma": 1}, "products": {"Coherence": 1}},
        "k": 0.35 * tau,
        "k_rev": 0.05,
    },
    {
        "label": "Storge protection",
        "eq": {
            "reactants": {"Storge": 1},
            "products": {"Coherence": 1, "Entropy": -0.5},
        },
        "k": 0.3 * tau,
        "k_rev": 0.02,
    },
    {
        "label": "Fear contraction",
        "eq": {
            "reactants": {"Fear": 1, "Openness": 1, "Coherence": 1},
            "products": {"Entropy": 2},
        },
        "k": 0.85 * tau,
        "k_rev": 0.0,
    },
    {
        "label": "Sadness reset",
        "eq": {
            "reactants": {"Sadness": 1, "Entropy": 0.5},
            "products": {"Coherence": 0.5},
        },
        "k": 0.25 * tau,
        "k_rev": 0.01,
    },
    {
        "label": "Awe transcendence",
        "eq": {"reactants": {"Awe": 1}, "products": {"Openness": 1, "Coherence": 0.5}},
        "k": 0.6 * tau,
        "k_rev": 0.03,
    },
    {
        "label": "Joy→Openness",
        "eq": {"reactants": {"Joy": 1}, "products": {"Openness": 1}},
        "k": 0.45 * tau,
        "k_rev": 0.03,
    },
    {
        "label": "Curiosity→Openness",
        "eq": {"reactants": {"Curiosity": 1}, "products": {"Openness": 1}},
        "k": 0.5 * tau,
        "k_rev": 0.03,
    },
    {
        "label": "Hope keeps doors open",
        "eq": {
            "reactants": {"Hope": 1},
            "products": {"Openness": 0.5, "Coherence": 0.5},
        },
        "k": 0.35 * tau,
        "k_rev": 0.02,
    },
]

# -----------------------------
# 3) Build stoichiometric matrix
# -----------------------------
idx = {s: i for i, s in enumerate(SPECIES)}


def net_change(reaction):
    dc = [0.0] * len(SPECIES)
    R = reaction["eq"]["reactants"]
    P = reaction["eq"]["products"]
    for r, coef in R.items():
        dc[idx[r]] -= coef
    for p, coef in P.items():
        dc[idx[p]] += coef
    return dc


def rate(x, reaction):
    v_f = reaction["k"]
    for r, coef in reaction["eq"]["reactants"].items():
        v_f *= max(0.0, x[idx[r]]) ** coef
    k_rev = reaction.get("k_rev", 0.0)
    v_r = 0.0
    if k_rev > 0:
        v_r = k_rev
        for p, coef in reaction["eq"]["products"].items():
            if coef > 0:
                v_r *= max(0.0, x[idx[p]]) ** abs(coef)
    return v_f - v_r


NET = [net_change(r) for r in REACTIONS]

# -----------------------------
# 4) CCI mapping coefficients
# -----------------------------
VALENCE = defaultdict(lambda: (1.0, 1.0, 1.0, 1.0))
boost = {
    "Agape": (1.05, 1.08, 1.03, 0.95),
    "Compassion": (1.03, 1.07, 1.02, 0.95),
    "Gratitude": (1.02, 1.06, 1.01, 0.96),
    "Awe": (1.04, 1.05, 1.08, 0.94),
    "Joy": (1.02, 1.04, 1.04, 0.97),
    "Curiosity": (1.01, 1.02, 1.06, 0.98),
    "Philia": (1.01, 1.04, 1.00, 0.98),
    "Pragma": (1.02, 1.03, 0.99, 0.98),
    "Ludus": (0.99, 0.98, 1.04, 1.01),
    "Eros": (0.98, 0.97, 1.06, 1.04),
    "Anger": (0.95, 0.93, 1.04, 1.08),
    "Fear": (0.90, 0.92, 0.92, 1.12),
    "Sadness": (0.97, 1.01, 0.96, 1.03),
    "Hope": (1.01, 1.02, 1.02, 0.98),
}
for k, v in boost.items():
    VALENCE[k] = v

BASE_CCI = {"cal": 0.82, "coh": 0.85, "em": 0.80, "noise": 0.18}


def cci_from_state(xvec):
    cal = BASE_CCI["cal"]
    coh = BASE_CCI["coh"]
    em = BASE_CCI["em"]
    nz = BASE_CCI["noise"]
    for s in SPECIES:
        a, b, c, d = VALENCE[s]
        conc = max(0.0, xvec[idx[s]])
        cal *= 1 + (a - 1) * conc
        coh *= 1 + (b - 1) * conc
        em *= 1 + (c - 1) * conc
        nz *= max(0.05, 1 + (d - 1) * conc)
    return (cal * coh * em) / max(0.01, nz)


# -----------------------------
# 5) Simulate
# -----------------------------
x = [x0[s] for s in SPECIES]
rows = []


def clamp01(z):
    return max(0.0, min(1.5, z))


for t in range(T):
    cci_val = cci_from_state(x)
    row = {"t": round(t * dt, 5), "CCI": cci_val}
    for s in SPECIES:
        row[s] = x[idx[s]]
    rows.append(row)

    dx = [0.0] * len(SPECIES)
    for j, r in enumerate(REACTIONS):
        v = rate(x, r)
        for i in range(len(SPECIES)):
            dx[i] += NET[j][i] * v
    coh = x[idx["Coherence"]]
    ent = x[idx["Entropy"]]
    dx[idx["Entropy"]] += 0.2 * (ent - 0.5 * coh) - leak * ent
    dx[idx["Coherence"]] += 0.2 * (coh - 0.3 * ent) - leak * coh
    dx[idx["Openness"]] += -leak * x[idx["Openness"]]

    for i in range(len(SPECIES)):
        x[i] = clamp01(x[i] + dt * dx[i])

# -----------------------------
# 6) Export CSVs (prefer /mnt, else local)
# -----------------------------
preferred = Path("/mnt")
if preferred.exists() and os.access(preferred, os.W_OK):
    out_dir = preferred / "data"
else:
    out_dir = Path("data")
out_dir.mkdir(parents=True, exist_ok=True)

traj_path = out_dir / "emostoch_trajectories.csv"
with open(traj_path, "w", newline="") as f:
    cols = ["t", "CCI"] + SPECIES
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)

summary_path = out_dir / "emostoch_summary.csv"
with open(summary_path, "w", newline="") as f:
    cols = ["species", "start", "end", "delta"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for s in SPECIES:
        start = rows[0][s]
        end = rows[-1][s]
        w.writerow({"species": s, "start": start, "end": end, "delta": end - start})

# Also copy to discovery_results for record
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dr = Path("discovery_results") / f"emostoch_{stamp}"
dr.mkdir(parents=True, exist_ok=True)
with open(dr / "emostoch_trajectories.csv", "w", newline="") as f:
    cols = ["t", "CCI"] + SPECIES
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow(r)
with open(dr / "emostoch_summary.csv", "w", newline="") as f:
    cols = ["species", "start", "end", "delta"]
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for s in SPECIES:
        start = rows[0][s]
        end = rows[-1][s]
        w.writerow({"species": s, "start": start, "end": end, "delta": end - start})

print(
    f"Exported:\n - {traj_path}\n - {summary_path}\nDiscovery copy:\n - {dr / 'emostoch_trajectories.csv'}\n - {dr / 'emostoch_summary.csv'}"
)
