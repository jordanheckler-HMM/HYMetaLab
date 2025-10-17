#!/usr/bin/env python3
"""
Panspermia Phase V runner
Cyclic openness pulses and feedback-driven coupling; computes EOI, FFT, coherence, phase-lock, and shock resync.
"""
import datetime
import hashlib
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

ROOT = Path(".")
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
DATA.mkdir(exist_ok=True)
PLOTS.mkdir(exist_ok=True)

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "discovery_results" / f"panspermia_phaseV_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

# --- Config defaults ---
EPOCHS = 12000
SEEDS = [1, 2, 3, 4]
LOG_EVERY = 10
STABILITY_WINDOW = 400
BASE_EPSILON = 0.0015
BASE_LAMBDA = 1e-5

PULSE = {"amplitude": 8.0, "width": 50, "period": 1000, "shape": "gaussian"}  # ×8

COUPLING = {
    "mode": "entropy_linked",
    "strength": 0.25,
}  # fixed | entropy_linked | cci_linked

SHOCK_ENABLED = True
T_SHOCK = 6000
SHOCK_INTENSITY = 0.35
SHOCK_DURATION = 25

MODES = [
    "baseline_closed",
    "cyclic_fixed",
    "cyclic_entropy_linked",
    "cyclic_cci_linked",
]

# anchors
earth_file = DATA / "earth_bench.json"
if earth_file.exists():
    earth = json.loads(earth_file.read_text())
else:
    earth = {
        "t_origin_Gyr": 4.5,
        "code_opt_z": 2.5,
        "homochirality_lock_in_score": 0.8,
        "entropy_rate_CMB": 1.0,
        "dark_energy_density": 0.69,
    }
v_astro = float(earth.get("entropy_rate_CMB", 1.0)) * float(
    earth.get("dark_energy_density", 1.0)
)
ASTRO_XMAX = max(v_astro, 1.0)
astro_anchor = v_astro / ASTRO_XMAX


def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


bio_anchor = logistic(
    0.6 * float(earth.get("code_opt_z", 2.5))
    + 0.3 * (1.0 / float(earth.get("t_origin_Gyr", 4.5)))
    + 0.1 * float(earth.get("homochirality_lock_in_score", 0.8))
)

print("Phase V anchors", astro_anchor, bio_anchor)

# helpers
EPS = 1e-12


def clamp01(arr):
    return np.minimum(np.maximum(arr, 0.0), 1.0)


# analytic signal via FFT (to compute instantaneous phase)
def analytic_signal(x):
    X = np.fft.fft(x)
    N = len(X)
    H = np.zeros(N)
    if N % 2 == 0:
        H[0] = 1
        H[1 : N // 2] = 2
        H[N // 2] = 1
    else:
        H[0] = 1
        H[1 : (N + 1) // 2] = 2
    Xa = X * H
    xa = np.fft.ifft(Xa)
    return xa


# FFT spectrum helper
def compute_spectrum(x, dt=1.0):
    # x: 1D array, dt: time step per sample (epochs between samples)
    n = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(X) ** 2
    return freqs, power, X


# containers
runs = []
eoi_times = []
pulse_events = []

# precompute pulse shape
width = PULSE["width"]
period = PULSE["period"]
amplitude = PULSE["amplitude"]
shape = PULSE["shape"]
sigma = width / 2.0

pulse_centers = list(range(0, EPOCHS, period))

print("Pulse centers sample:", pulse_centers[:5])

# simulate per mode and seed
for mode in MODES:
    print("Running mode", mode)
    for seed in SEEDS:
        rng = np.random.RandomState(int(seed))
        agents = me.initialize_agents(200, 3, 0.05, rng)
        epochs_logged = []
        cci_logged = []
        hazard_logged = []
        eps_logged = []
        entropy_logged = []
        eoi_logged = []
        cci_max = 0.0

        for t in range(EPOCHS):
            # base epsilon
            eps0 = BASE_EPSILON
            lam = BASE_LAMBDA
            base = eps0 * math.exp(-lam * t)
            # pulse contribution
            pulse_val = 0.0
            for center in pulse_centers:
                if abs(t - center) > 4 * width:
                    continue
                if shape == "gaussian":
                    pulse_val += amplitude * math.exp(
                        -((t - center) ** 2) / (2 * sigma * sigma)
                    )
                elif shape == "square":
                    if abs(t - center) <= width / 2:
                        pulse_val += amplitude
                elif shape == "sinusoidal":
                    pulse_val += amplitude * math.sin(
                        2 * math.pi * (t - center) / width
                    )
            # multiplicative factor
            if mode == "baseline_closed":
                eps_eff = base
            else:
                eps_eff = base * (1.0 + pulse_val)
            # coupling adjustments
            # compute entropy and CCI quickly from current agents
            alive = [a for a in agents if a["alive"]]
            B = np.array([a["belief"] for a in agents]) if agents else np.array([])
            mean_b = B.mean(axis=0) if B.size else np.array([])
            if mean_b.size:
                S = -np.sum(mean_b * np.log(mean_b + 1e-12))
                S_max = math.log(len(mean_b) + 1e-12)
            else:
                S = 0.0
                S_max = 1.0
            cci_now = me.collective_cci(agents)
            if mode == "cyclic_entropy_linked":
                f_ent = COUPLING["strength"] * (1.0 - (S / (S_max + 1e-12)))
                eps_eff = eps_eff * (1.0 + f_ent)
            if mode == "cyclic_cci_linked":
                # need cci_max, approximate with running max
                # use cci_max to scale
                if cci_max <= 0:
                    cci_scale = 0.0
                else:
                    cci_scale = cci_now / (cci_max + EPS)
                f_cci = COUPLING["strength"] * cci_scale
                eps_eff = eps_eff * (1.0 + f_cci)

            # shock
            if SHOCK_ENABLED and T_SHOCK <= t < T_SHOCK + SHOCK_DURATION:
                current_shock = SHOCK_INTENSITY
                for a in agents:
                    if a["alive"]:
                        a["resource"] -= SHOCK_INTENSITY * 0.2
                        if a["resource"] < 0:
                            a["alive"] = False
            else:
                current_shock = 0.0

            # openness inflow
            if eps_eff > 0:
                alive = [a for a in agents if a["alive"]]
                if alive:
                    for a in alive:
                        a["resource"] = min(1.0, a["resource"] + eps_eff)
            # update
            me.step_update(agents, current_shock, "chronic", rng)
            # metrics
            alive = [a for a in agents if a["alive"]]
            hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(len(agents))
            cci = me.collective_cci(agents)
            cci_max = max(cci_max, cci)
            # entropy
            B = np.array([a["belief"] for a in agents]) if agents else np.array([])
            mean_b = B.mean(axis=0) if B.size else np.array([])
            if mean_b.size:
                S = -np.sum(mean_b * np.log(mean_b + 1e-12))
            else:
                S = 0.0

            if t % LOG_EVERY == 0:
                epochs_logged.append(t)
                cci_logged.append(cci)
                hazard_logged.append(hazard)
                eps_logged.append(eps_eff)
                entropy_logged.append(S)

        # compute collapse_risk_norm
        h_arr = np.array(hazard_logged)
        h_min = float(np.min(h_arr)) if h_arr.size else 0.0
        h_max = float(np.max(h_arr)) if h_arr.size else h_min
        denom = max(1e-9, h_max - h_min)
        cr_norm = clamp01((h_arr - h_min) / denom)
        eps_arr = np.array(eps_logged)
        cum_mean_eps = (
            np.array([eps_arr[: i + 1].mean() for i in range(len(eps_arr))])
            if len(eps_arr) > 0
            else np.array([])
        )
        # compute EOI
        eoi_arr = []
        for i in range(len(epochs_logged)):
            cci_val = cci_logged[i]
            cci_scale = (cci_val / (cci_max + EPS)) if cci_max > 0 else 0.0
            eoi = (
                cum_mean_eps[i]
                * (1.0 - cr_norm[i])
                * cci_scale
                * astro_anchor
                * bio_anchor
            )
            eoi_arr.append(float(eoi))
        eoi_arr = clamp01(np.array(eoi_arr))

        # record per-epoch
        for i, t in enumerate(epochs_logged):
            eoi_times.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "epoch": t,
                    "cci": cci_logged[i],
                    "hazard_raw": hazard_logged[i],
                    "collapse_risk_norm": float(cr_norm[i]),
                    "eps_eff": float(eps_logged[i]),
                    "entropy": float(entropy_logged[i]),
                    "EOI": float(eoi_arr[i]),
                }
            )
        # pulse event logging: mark pulse centers that affected this run
        for center in pulse_centers:
            start = center - width
            end = center + width
            if start < 0 or start >= EPOCHS:
                continue
            # compute example eps at center
            eps_center = BASE_EPSILON * math.exp(-BASE_LAMBDA * center)
            pulse_events.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "center": center,
                    "start": start,
                    "end": end,
                    "amplitude": amplitude,
                    "shape": shape,
                    "eps_center": eps_center,
                }
            )

        # spectral analysis on EOI time series
        # use eoi_arr (logged every LOG_EVERY epochs). dt = LOG_EVERY
        dt = LOG_EVERY
        if len(eoi_arr) >= 8:
            freqs, power, Xc = compute_spectrum(eoi_arr, dt=dt)
            # ignore zero freq
            if freqs.size > 1:
                idx = np.argmax(power[1:]) + 1
                fstar = freqs[idx]
                power_f = power[idx]
                power_bg = np.mean(np.delete(power, idx))
                coherence_ratio = float(power_f / (power_bg + EPS))
                # analytic signal phases for phase-lock
                xa = analytic_signal(eoi_arr)
                phase_eoi = np.angle(xa)
                # entropy phase
                ent = np.array(entropy_logged)
                if len(ent) == len(eoi_arr):
                    xa_ent = analytic_signal(ent)
                    phase_ent = np.angle(xa_ent)
                    # phase-lock index: mean(cos(delta_phase))
                    delta = phase_eoi - phase_ent
                    phase_lock = float(np.mean(np.cos(delta)))
                else:
                    phase_lock = float("nan")
            else:
                fstar = float("nan")
                coherence_ratio = float("nan")
                phase_lock = float("nan")
        else:
            fstar = float("nan")
            coherence_ratio = float("nan")
            phase_lock = float("nan")

        # shock resync: compute phase before shock and after shock, time to resync
        # compute analytic phase over whole eoi_arr
        xa_full = analytic_signal(eoi_arr) if len(eoi_arr) else np.array([])
        phase_full = np.angle(xa_full) if xa_full.size else np.array([])
        # find index of epoch nearest T_SHOCK in epochs_logged
        shock_idx = None
        for i, t in enumerate(epochs_logged):
            if t >= T_SHOCK:
                shock_idx = i
                break
        resync_time = float("nan")
        delta_phase = float("nan")
        if shock_idx is not None and phase_full.size:
            # baseline phase = mean phase over window before shock (100 epochs)
            pre_start = max(0, shock_idx - (100 // LOG_EVERY))
            pre_phase = phase_full[pre_start:shock_idx]
            post_phase = (
                phase_full[shock_idx : shock_idx + (200 // LOG_EVERY)]
                if shock_idx + (200 // LOG_EVERY) < len(phase_full)
                else phase_full[shock_idx:]
            )
            if pre_phase.size and post_phase.size:
                mean_pre = np.angle(np.mean(np.exp(1j * pre_phase)))
                # find time until phase returns to within 10 degrees (~0.1745 rad) of mean_pre
                for j in range(shock_idx, len(phase_full)):
                    cur = phase_full[j]
                    d = abs(((cur - mean_pre + math.pi) % (2 * math.pi)) - math.pi)
                    if d <= math.radians(10.0):
                        resync_time = float(epochs_logged[j] - T_SHOCK)
                        break
                delta_phase = float(
                    ((phase_full[shock_idx] - mean_pre + math.pi) % (2 * math.pi))
                    - math.pi
                )
        # per-run summary
        runs.append(
            {
                "mode": mode,
                "seed": seed,
                "mean_EOI": float(np.mean(eoi_arr)) if len(eoi_arr) else 0.0,
                "coherence_ratio": coherence_ratio,
                "dominant_f": float(fstar),
                "phase_lock": phase_lock,
                "resync_time": resync_time,
                "delta_phase": delta_phase,
                "delta_CCI_postshock": (
                    float(delta_CCI_post)
                    if "delta_CCI_post" in locals()
                    else float("nan")
                ),
            }
        )

# write outputs
pd.DataFrame(runs).to_csv(DATA / "runs_phaseV_summary.csv", index=False)
pd.DataFrame(pulse_events).to_csv(DATA / "pulse_events.csv", index=False)
pd.DataFrame(eoi_times).to_csv(DATA / "eoi_cycles.csv", index=False)
# frequency spectrum: flatten per-run? create simple summary from runs
freq_df = pd.DataFrame(
    [
        {
            "mode": r["mode"],
            "seed": r["seed"],
            "fstar": r["dominant_f"],
            "coherence_ratio": r["coherence_ratio"],
        }
        for r in runs
    ]
)
freq_df.to_csv(DATA / "frequency_spectrum.csv", index=False)
pd.DataFrame(
    [
        {"mode": r["mode"], "seed": r["seed"], "phase_lock": r["phase_lock"]}
        for r in runs
    ]
).to_csv(DATA / "phase_lock_metrics.csv", index=False)

# Bayes update: L_cycle = P(coherence_ratio > 1.5 | cyclic modes)/P(...|baseline)
rr = pd.DataFrame(runs)
base_prop = (
    rr[(rr["mode"] == "baseline_closed") & (rr["coherence_ratio"] > 1.5)].shape[0]
) / max(1, rr[rr["mode"] == "baseline_closed"].shape[0])
cyclic_prop = (
    rr[(rr["mode"] != "baseline_closed") & (rr["coherence_ratio"] > 1.5)].shape[0]
) / max(1, rr[rr["mode"] != "baseline_closed"].shape[0])
L_cycle = (cyclic_prop + 1e-6) / (base_prop + 1e-6)
# combine with Phase IV bayes if exists
pv4 = DATA / "bayes_phaseIV.json"
if pv4.exists():
    prev = json.loads(pv4.read_text())
    prev_bf = prev.get("combined_bf", 1.0)
else:
    prev_bf = 1.0
combined_bf = float(prev_bf * L_cycle)
with open(DATA / "bayes_phaseV.json", "w") as f:
    json.dump(
        {"L_cycle": L_cycle, "combined_bf": combined_bf, "prev_bf": prev_bf},
        f,
        indent=2,
    )

# plots
rdf = pd.DataFrame(runs)
if not rdf.empty:
    # coherence_ratio vs mode
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(MODES):
        sub = rdf[rdf["mode"] == m]
        if sub.empty:
            continue
        x = np.full(len(sub), i)
        plt.scatter(x, sub["coherence_ratio"], alpha=0.6)
    plt.xticks(range(len(MODES)), MODES)
    plt.ylabel("coherence_ratio")
    plt.title("Coherence ratio by mode")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "coherence_ratio_vs_mode.png")
    plt.close()

# eoi timeseries example
edf = pd.DataFrame(eoi_times)
if not edf.empty:
    plt.figure(figsize=(8, 4))
    for m in MODES:
        sel = edf[edf["mode"] == m]
        if sel.empty:
            continue
        grp = sel.groupby("epoch")["EOI"].mean()
        plt.plot(grp.index, grp.values, label=m)
    plt.legend()
    plt.title("EOI cycles (mean across seeds)")
    plt.xlabel("epoch")
    plt.ylabel("EOI")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "eoi_cycles_timeseries.png")
    plt.close()

# FFT spectrum (aggregate)
if not edf.empty:
    plt.figure(figsize=(6, 4))
    for m in MODES:
        sel = edf[edf["mode"] == m]
        if sel.empty:
            continue
        # take mean EOI time series and compute spectrum
        grp = sel.groupby("epoch")["EOI"].mean().values
        if len(grp) < 8:
            continue
        freqs, power, _ = compute_spectrum(grp, dt=LOG_EVERY)
        plt.plot(freqs[1:], power[1:], label=m)
    plt.legend()
    plt.xlabel("frequency (1/epochs)")
    plt.ylabel("power")
    plt.title("FFT spectra by mode")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "fft_spectrum.png")
    plt.close()

# phase lock heatmap (mode x seed average)
plm = pd.DataFrame(
    [
        {"mode": r["mode"], "seed": r["seed"], "phase_lock": r["phase_lock"]}
        for r in runs
    ]
)
if not plm.empty:
    pivot = plm.pivot_table(index="mode", columns="seed", values="phase_lock")
    plt.figure(figsize=(6, 4))
    plt.imshow(pivot.fillna(0).values, aspect="auto", cmap="RdYlBu", vmin=-1, vmax=1)
    plt.colorbar(label="phase_lock")
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("seed")
    plt.title("Phase-lock heatmap")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "phase_lock_heatmap.png")
    plt.close()

# shock resync plot: mean resync time by mode
if not rdf.empty:
    plt.figure(figsize=(6, 4))
    for i, m in enumerate(MODES):
        sub = rdf[rdf["mode"] == m]
        if sub.empty:
            continue
        # plot only non-null resync_time
        vals = sub["resync_time"].dropna()
        if vals.size == 0:
            continue
        x = np.full(len(vals), i)
        plt.scatter(x, vals, alpha=0.6)
    plt.xticks(range(len(MODES)), MODES)
    plt.ylabel("resync_time (epochs)")
    plt.title("Shock resync by mode")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "shock_resync.png")
    plt.close()

# save outputs and bundle
pd.DataFrame(runs).to_csv(OUT / "runs_phaseV_summary.csv", index=False)
pd.DataFrame(pulse_events).to_csv(DATA / "pulse_events.csv", index=False)
pd.DataFrame(eoi_times).to_csv(DATA / "eoi_cycles.csv", index=False)
freq_df.to_csv(DATA / "frequency_spectrum.csv", index=False)
plm.to_csv(DATA / "phase_lock_metrics.csv", index=False)

bundle = OUT / f"panspermia_phaseV_bundle_{STAMP}.zip"
with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
    for f in (
        list(OUT.rglob("*")) + list(DATA.glob("*.csv")) + list(DATA.glob("*.json"))
    ):
        if f.is_file():
            z.write(f, arcname=str(f.relative_to(ROOT)))
h = hashlib.sha256()
with open(bundle, "rb") as bf:
    for chunk in iter(lambda: bf.read(1 << 20), b""):
        h.update(chunk)
with open(OUT / "SHA256SUMS.txt", "w") as s:
    s.write(f"{h.hexdigest()}  {bundle.name}\n")

print("Phase V complete. Outputs in", OUT)
print("Data in", DATA)
print("Bundle:", bundle)
