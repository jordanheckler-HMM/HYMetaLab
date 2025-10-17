import json
import math
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------
# Phase 23 — Wormhole Coupling Test
# -----------------------------
# Two universes (A & B) with the same base params.
# B gets a shock; a wormhole A->B opens for a short window to share "structured information"
# gated by A's coherence (CCI). We log trajectories and compare recovery & hazard.

# ---------- Experiment knobs (edit freely) ----------
T = 1200  # total epochs
seed = 42  # RNG seed for repeatability
n_agents = 180  # agent count (only affects scaling heuristics here)
eps = 0.006  # openness (epsilon) for both universes
lam = 0.90  # temporal feedback strength (lambda)
base_energy = 1.0  # normalized energy inflow scale
base_info = 1.0  # normalized info flux scale
coord = 0.55  # coordination level (0-1)
ineq = 0.22  # inequality proxy (0-1)

# Shock to B
shock_t0 = 520  # start epoch for shock
shock_t1 = 540  # end epoch (inclusive)
shock_noise = 0.30  # noise increment during shock window

# Wormhole window (A -> B)  **changed to overlap shock + early recovery**
wh_t0 = 520  # start during shock
wh_t1 = 580  # continue through early recovery
wh_coupling = 0.10  # stronger bridge so effect is visible
wh_gate_cci = 0.58  # lower gate so A actually qualifies
wh_energy_cost = 0.05  # cheaper to send info so A doesn’t starve

# Output
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = f"discovery_results/phase23_wormhole_{stamp}"
os.makedirs(outdir, exist_ok=True)

np.random.seed(seed)


# ---------- Helper dynamics ----------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def update_system(state, params, ex_noise=0.0, import_info=0.0, energy_cost=0.0):
    """
    Minimal phenomenological update that respects project logic:
    - CCI rises with (energy * info * coord) and lam feedback; falls with noise & inequality
    - hazard is the moving 'risk' proxy: rises with noise & inequality; falls with CCI & openness
    - survival is a soft-decreasing function of hazard
    """
    CCI, hazard, energy, info, survival, struct_info = state
    eps, lam, coord, ineq = (
        params["eps"],
        params["lam"],
        params["coord"],
        params["ineq"],
    )

    # base inflows (openness-buffered)
    energy_in = params["base_energy"] * (1 + 0.5 * eps)
    info_in = params["base_info"] * (1 + 0.5 * eps)

    # apply wormhole inputs (info import) and energy costs if any
    info_in += import_info
    energy -= energy_cost * abs(import_info)
    energy = max(0.0, energy)

    # effective "structured info" produced this step
    struct_info = info_in * (0.6 + 0.4 * coord) * (1 - 0.3 * ineq)

    # noise dynamics: external + endogenous from inequality/complexity
    noise = 0.08 + 0.6 * ineq + ex_noise

    # CCI integrates: prior memory (lam), plus fresh structured info modulated by energy & eps, minus noise
    cci_drive = (struct_info * (0.5 + 0.5 * energy) * (0.6 + 0.4 * eps)) - noise
    CCI_next = clamp(lam * CCI + (1 - lam) * sigmoid(cci_drive), 0.0, 1.0)

    # Hazard tilts opposite of CCI, damped by openness & coord, increased by noise
    hazard_drive = noise * (1.0 + 0.4 * ineq) - 0.6 * CCI_next - 0.2 * coord - 0.2 * eps
    hazard_next = clamp(0.55 * hazard + 0.45 * sigmoid(hazard_drive), 0.0, 1.0)

    # Survival proxy (soft): higher hazard => lower survival
    survival = clamp(1.0 - 0.8 * hazard_next, 0.0, 1.0)

    # Update crude stores
    energy_next = clamp(energy + energy_in - 0.05, 0.0, 2.0)
    info_next = clamp(info + info_in - 0.05, 0.0, 2.0)

    return (CCI_next, hazard_next, energy_next, info_next, survival, struct_info)


def wormhole_transfer(A_state, params):
    """Compute info transfer from A to B gated by CCI and scaled by coupling."""
    CCI_A, _, _, _, _, struct_info_A = A_state
    if CCI_A >= wh_gate_cci:
        transfer = wh_coupling * struct_info_A
        energy_cost = wh_energy_cost * transfer
        return transfer, energy_cost
    return 0.0, 0.0


# ---------- Run the experiment ----------
def run_pair(with_wormhole=True):
    # initial states
    A = (
        0.55,
        0.20,
        1.00,
        1.00,
        1.0,
        0.0,
    )  # (CCI, hazard, energy, info, survival, struct_info)
    B = (0.55, 0.20, 1.00, 1.00, 1.0, 0.0)

    params = {
        "eps": eps,
        "lam": lam,
        "coord": coord,
        "ineq": ineq,
        "base_energy": base_energy,
        "base_info": base_info,
    }

    traj = []
    for t in range(T):
        # external shock noise to B in [shock_t0, shock_t1]
        ex_noise_B = shock_noise if (shock_t0 <= t <= shock_t1) else 0.0

        # optional wormhole window
        imp_B, cost_A = (0.0, 0.0)
        if with_wormhole and (wh_t0 <= t <= wh_t1):
            imp_B, cost_A = wormhole_transfer(A, params)

        A = update_system(A, params, ex_noise=0.0, import_info=0.0, energy_cost=cost_A)
        B = update_system(
            B, params, ex_noise=ex_noise_B, import_info=imp_B, energy_cost=0.0
        )

        traj.append(
            {
                "t": t,
                "seed": seed,
                "n_agents": n_agents,
                "eps": eps,
                "lam": lam,
                "coord": coord,
                "ineq": ineq,
                "shock_window": int(shock_t0 <= t <= shock_t1),
                "wormhole_window": int(with_wormhole and (wh_t0 <= t <= wh_t1)),
                # Universe A
                "A_CCI": A[0],
                "A_hazard": A[1],
                "A_energy": A[2],
                "A_info": A[3],
                "A_survival": A[4],
                # Universe B
                "B_CCI": B[0],
                "B_hazard": B[1],
                "B_energy": B[2],
                "B_info": B[3],
                "B_survival": B[4],
            }
        )

    df = pd.DataFrame(traj)
    return df


# Baseline pair (no wormhole), then with wormhole
df_nowh = run_pair(with_wormhole=False)
df_wh = run_pair(with_wormhole=True)


# ---------- Metrics ----------
def window_mean(x, t0, t1):
    m = x[(x.index >= t0) & (x.index <= t1)].mean()
    return float(m) if not math.isnan(m) else float("nan")


def compute_summary(df, label):
    # Recovery time after shock: when B_CCI regains 95% of its pre-shock mean
    pre_mean = df.loc[df["t"] < shock_t0, "B_CCI"].mean()
    target = 0.95 * pre_mean
    rec_t = None
    for t in range(shock_t1 + 1, T):
        if df.loc[df["t"] == t, "B_CCI"].values[0] >= target:
            rec_t = t - (shock_t1 + 1)
            break

    # AUH in a window around the shock (area-under-hazard)
    around0, around1 = shock_t0 - 20, shock_t1 + 80
    sub = df[(df["t"] >= around0) & (df["t"] <= around1)].copy()
    AUH = float(np.trapz(sub["B_hazard"], sub["t"]))

    # Stability window (last 200 epochs)
    stable = df[df["t"] >= T - 200]
    summary = {
        "label": label,
        "B_stability_CCI_mean": float(stable["B_CCI"].mean()),
        "B_stability_hazard_mean": float(stable["B_hazard"].mean()),
        "B_recovery_time_after_shock": None if rec_t is None else int(rec_t),
        "B_AUH_shock_window": AUH,
        "A_stability_CCI_mean": float(stable["A_CCI"].mean()),
        "A_stability_hazard_mean": float(stable["A_hazard"].mean()),
    }
    return summary


sum_nowh = compute_summary(df_nowh, "no_wormhole")
sum_wh = compute_summary(df_wh, "wormhole")

summary_df = pd.DataFrame([sum_nowh, sum_wh])

# ---------- Exports ----------
# Data
df_nowh.to_csv(os.path.join(outdir, "trajectories_no_wormhole.csv"), index=False)
df_wh.to_csv(os.path.join(outdir, "trajectories_wormhole.csv"), index=False)
summary_df.to_csv(os.path.join(outdir, "summary.csv"), index=False)

# JSON bundle
with open(os.path.join(outdir, "summary.json"), "w") as f:
    json.dump(
        {
            "params": {
                "T": T,
                "seed": seed,
                "n_agents": n_agents,
                "eps": eps,
                "lam": lam,
                "coord": coord,
                "ineq": ineq,
                "shock": [shock_t0, shock_t1, shock_noise],
                "wormhole": [wh_t0, wh_t1, wh_coupling, wh_gate_cci, wh_energy_cost],
            },
            "results": summary_df.to_dict(orient="records"),
        },
        f,
        indent=2,
    )

# Figures
plt.figure(figsize=(10, 5))
plt.plot(df_nowh["t"], df_nowh["B_CCI"], label="B_CCI (no wormhole)")
plt.plot(df_wh["t"], df_wh["B_CCI"], label="B_CCI (+ wormhole)")
plt.axvspan(shock_t0, shock_t1, color="grey", alpha=0.2, label="Shock window")
plt.axvspan(wh_t0, wh_t1, color="cyan", alpha=0.15, label="Wormhole")
plt.xlabel("Epoch")
plt.ylabel("CCI (Universe B)")
plt.title("Universe B — CCI with vs without Wormhole")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "B_CCI_compare.png"), dpi=160)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df_nowh["t"], df_nowh["B_hazard"], label="B_hazard (no wormhole)")
plt.plot(df_wh["t"], df_wh["B_hazard"], label="B_hazard (+ wormhole)")
plt.axvspan(shock_t0, shock_t1, color="grey", alpha=0.2, label="Shock window")
plt.axvspan(wh_t0, wh_t1, color="cyan", alpha=0.15, label="Wormhole")
plt.xlabel("Epoch")
plt.ylabel("Hazard (Universe B)")
plt.title("Universe B — Hazard with vs without Wormhole")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "B_hazard_compare.png"), dpi=160)
plt.close()

# Markdown report
with open(os.path.join(outdir, "report.md"), "w") as f:
    f.write("# Phase 23 — Wormhole Coupling Test\n\n")
    f.write(
        "**Setup:** Two identical universes; B receives a noise shock; optional wormhole (A→B) shares structured information gated by CCI.\n\n"
    )
    f.write("**Key params:**\n")
    f.write(f"- eps={eps}, lam={lam}, coord={coord}, ineq={ineq}\n")
    f.write(
        f"- shock=[{shock_t0}, {shock_t1}, {shock_noise}], wormhole=[{wh_t0}, {wh_t1}, {wh_coupling}, gate={wh_gate_cci}, cost={wh_energy_cost}]\n\n"
    )
    f.write("**Results (Universe B):**\n\n")
    f.write(summary_df.to_markdown(index=False))
    f.write("\n\n**Figures:** `B_CCI_compare.png`, `B_hazard_compare.png`\n")

print(f"✅ Done. See: {outdir}")
