#!/usr/bin/env python3
"""
Panspermia Phase VI runner
Resonance sweep for pulsed openness and feedback lag to find persistent heartbeat
Defaults to a sampled subset to keep interactive runs fast; use --full to run entire grid offline.
"""
import argparse
import datetime
import hashlib
import json
import math
import random
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import meaning_experiment as me

# ---------- Configuration (defaults from user request) ----------
DEFAULTS = {
    "epochs": 14000,
    "seeds": [1, 2, 3, 4],
    "log_every": 10,
    "stability_window_last": 400,
    "base_epsilon": 0.0015,
    "base_lambda_grid": [1e-5, 7.5e-6, 5e-6],
    "drive": {
        "amplitudes": [8, 12, 16],
        "periods": [500, 700, 900, 1200, 1600, 2000, 2500, 3000],
        "width_ratio": 0.06,
        "shape": "gaussian",
    },
    "feedback": {
        "coupling_mode": "entropy_linked",
        "strength_grid": [0.2, 0.35, 0.5],
        "lag_grid": [0, 10, 25, 50, 75, 100],
    },
    "shock_enabled": True,
    "shock": {"t_shock": 7000, "intensity": 0.35, "duration": 25},
    "settle_phase": {"t_on": 2000, "t_off": 11000},
}

ROOT = Path(".")
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
OUT = ROOT / "discovery_results"
for d in (DATA, PLOTS, OUT):
    d.mkdir(exist_ok=True)

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT / f"panspermia_phaseVI_{STAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--full", action="store_true", help="Run full grid (may be very slow)"
    )
    p.add_argument("--seed", type=int, default=0, help="random seed for sampling")
    return p.parse_args()


# ---------- Helpers (pulse envelope, analytic signal, spectrum, autocorr) ----------
def pulse_envelope(delta_t, period, width, shape="gaussian"):
    # delta_t: time relative to pulse center
    # width here is full width; for gaussian we treat sigma = width/2
    if shape == "gaussian":
        sigma = width / 2.0
        return math.exp(-((delta_t) ** 2) / (2 * sigma * sigma))
    elif shape == "square":
        return 1.0 if abs(delta_t) <= width / 2.0 else 0.0
    elif shape == "sinusoidal":
        # treat width as period for sinusoidal oscillation around center
        return 0.5 * (1 + math.sin(2 * math.pi * delta_t / width))
    else:
        return 0.0


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


def compute_spectrum(x, dt=1.0):
    n = len(x)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=dt)
    power = np.abs(X) ** 2
    return freqs, power, X


def autocorr_peak(x):
    # normalized autocorrelation, return max for lag>0
    x = np.asarray(x)
    if x.size < 2:
        return 0.0
    x = x - np.mean(x)
    norm = np.sum(x * x)
    if norm <= 0:
        return 0.0
    corr = np.correlate(x, x, mode="full")
    mid = len(corr) // 2
    ac = corr[mid + 1 :] / norm
    return float(np.max(ac)) if ac.size > 0 else 0.0


# ---------- Main runner ----------
def main():
    args = parse_args()
    rnd = random.Random(args.seed)
    cfg = DEFAULTS

    # load earth anchors
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

    print("Phase VI anchors", astro_anchor, bio_anchor)

    epochs = cfg["epochs"]
    seeds = cfg["seeds"]
    LOG_EVERY = cfg["log_every"]
    BASE_EPS = cfg["base_epsilon"]
    lam_grid = cfg["base_lambda_grid"]
    amps = cfg["drive"]["amplitudes"]
    periods = cfg["drive"]["periods"]
    width_ratio = cfg["drive"]["width_ratio"]
    shape = cfg["drive"]["shape"]
    coupling_mode = cfg["feedback"]["coupling_mode"]
    strengths = cfg["feedback"]["strength_grid"]
    lags = cfg["feedback"]["lag_grid"]
    shock_enabled = cfg["shock_enabled"]
    t_shock = cfg["shock"]["t_shock"]
    shock_int = cfg["shock"]["intensity"]
    shock_dur = cfg["shock"]["duration"]
    t_on = cfg["settle_phase"]["t_on"]
    t_off = cfg["settle_phase"]["t_off"]

    # Build grid
    grid = []
    for lam in lam_grid:
        for A in amps:
            for P in periods:
                for C in strengths:
                    for dt in lags:
                        grid.append((lam, A, P, C, dt))

    # If not full, sample a subset to keep runtime reasonable here
    if not args.full:
        max_samples = 24  # keep interactive run fast
        if len(grid) > max_samples:
            grid = rnd.sample(grid, max_samples)
            print(
                f"[INFO] Sampled {max_samples} parameter combos out of full {len(lam_grid)*len(amps)*len(periods)*len(strengths)*len(lags)}"
            )

    runs = []
    spectra = []
    pulse_events = []
    candidates = []

    for lam, A, P, C, dt_lag in grid:
        print("Running combo", lam, A, P, C, dt_lag)
        width = max(1, P * width_ratio)
        pulse_centers = list(range(t_on, t_off, P))

        for seed in seeds:
            rng = np.random.RandomState(int(seed + (int(lam * 1e9) % 100000)))
            agents = me.initialize_agents(200, 3, 0.05, rng)

            epochs_logged = []
            eoi_logged = []
            hazard_logged = []
            eps_logged = []
            cci_logged = []
            entropy_logged = []
            cci_max = 0.0

            # history buffers for feedback (store past S and CCI by epoch index)
            S_hist = []
            CCI_hist = []

            for t in range(epochs):
                # base epsilon
                base = BASE_EPS * math.exp(-lam * t)
                eps_eff = base

                # pulses only in window
                if t_on <= t < t_off:
                    env = 0.0
                    # sum contributions from pulses
                    for center in pulse_centers:
                        dtc = t - center
                        # limit to local window
                        if abs(dtc) > 4 * width:
                            continue
                        env += pulse_envelope(dtc, P, width, shape=shape)
                    if env > 0:
                        eps_pulse = base * (1.0 + A * env)
                    else:
                        eps_pulse = base
                    # feedback f(t) uses lagged S or CCI
                    f_mult = 1.0
                    if coupling_mode == "entropy_linked":
                        t_lag = t - dt_lag
                        if t_lag >= 0 and t_lag < len(S_hist):
                            S_lag = S_hist[t_lag]
                            S_max = math.log(max(2, len(agents)))
                            f_mult = 1.0 + C * (1.0 - (S_lag / (S_max + 1e-12)))
                    elif coupling_mode == "cci_linked":
                        t_lag = t - dt_lag
                        if t_lag >= 0 and t_lag < len(CCI_hist):
                            cci_lag = CCI_hist[t_lag]
                            f_mult = (
                                1.0 + C * (cci_lag / (cci_max + 1e-12))
                                if cci_max > 0
                                else 1.0
                            )
                    eps_eff = eps_pulse * f_mult

                # shock
                if shock_enabled and t_shock <= t < t_shock + shock_dur:
                    current_shock = shock_int
                    for a in agents:
                        if a["alive"]:
                            a["resource"] -= shock_int * 0.2
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

                me.step_update(agents, current_shock, "chronic", rng)

                # metrics
                hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(
                    len(agents)
                )
                cci = me.collective_cci(agents)
                cci_max = max(cci_max, cci)
                B = np.array([a["belief"] for a in agents]) if agents else np.array([])
                mean_b = B.mean(axis=0) if B.size else np.array([])
                if mean_b.size:
                    S = -np.sum(mean_b * np.log(mean_b + 1e-12))
                else:
                    S = 0.0

                # save history for lagged feedback
                S_hist.append(S)
                CCI_hist.append(cci)

                if t % LOG_EVERY == 0:
                    epochs_logged.append(t)
                    hazard_logged.append(hazard)
                    cci_logged.append(cci)
                    eps_logged.append(eps_eff)
                    entropy_logged.append(S)

            # post-run compute metrics
            h_arr = np.array(hazard_logged)
            h_min = float(np.min(h_arr)) if h_arr.size else 0.0
            h_max = float(np.max(h_arr)) if h_arr.size else h_min
            denom = max(1e-9, h_max - h_min)
            cr_norm = np.clip((h_arr - h_min) / denom, 0.0, 1.0)
            eps_arr = np.array(eps_logged)
            cum_mean_eps = (
                np.array([eps_arr[: i + 1].mean() for i in range(len(eps_arr))])
                if len(eps_arr) > 0
                else np.array([])
            )

            eoi_arr = []
            for i in range(len(epochs_logged)):
                cci_val = cci_logged[i]
                cci_scale = (cci_val / (cci_max + 1e-12)) if cci_max > 0 else 0.0
                eoi = (
                    cum_mean_eps[i]
                    * (1.0 - cr_norm[i])
                    * cci_scale
                    * astro_anchor
                    * bio_anchor
                )
                eoi_arr.append(float(np.clip(eoi, 0.0, 1.0)))
            eoi_arr = np.array(eoi_arr)

            # record per-epoch EOI rows
            for i, t in enumerate(epochs_logged):
                pulse_events.append(
                    {
                        "lambda": lam,
                        "amp": A,
                        "period": P,
                        "coupling": C,
                        "lag": dt_lag,
                        "seed": seed,
                        "epoch": t,
                        "eps_eff": float(eps_logged[i]),
                        "collapse_risk_norm": float(cr_norm[i]),
                        "cci": float(cci_logged[i]),
                        "entropy": float(entropy_logged[i]),
                        "EOI": float(eoi_arr[i]),
                    }
                )

            # spectral windows: need arrays corresponding to windows W1 W2 W3
            # convert epoch-index to indices in logged arrays
            def epoch_index(e):
                # return index in epochs_logged nearest to e
                arr = np.array(epochs_logged)
                if arr.size == 0:
                    return None
                idx = np.searchsorted(arr, e)
                if idx >= len(arr):
                    return len(arr) - 1
                return idx

            i_w1_start = epoch_index(t_on)
            i_w1_end = (
                epoch_index(t_shock) if t_shock is not None else epoch_index(t_off)
            )
            i_w2_start = epoch_index(t_shock + shock_dur)
            i_w2_end = epoch_index(t_off)
            i_w3_start = epoch_index(t_off)
            i_w3_end = len(eoi_arr) - 1

            def analyze_window(arr, i0, i1):
                if i0 is None or i1 is None or i1 - i0 < 4:
                    return {
                        "fstar": float("nan"),
                        "power_f": float("nan"),
                        "coherence": float("nan"),
                    }
                sub = arr[i0 : i1 + 1]
                if len(sub) < 8:
                    return {
                        "fstar": float("nan"),
                        "power_f": float("nan"),
                        "coherence": float("nan"),
                    }
                freqs, power, _ = compute_spectrum(sub, dt=LOG_EVERY)
                if freqs.size <= 1:
                    return {
                        "fstar": float("nan"),
                        "power_f": float("nan"),
                        "coherence": float("nan"),
                    }
                idx = np.argmax(power[1:]) + 1
                fstar = freqs[idx]
                power_f = float(power[idx])
                power_bg = float(np.median(np.delete(power, idx)))
                coherence = float(power_f / (power_bg + 1e-12))
                return {
                    "fstar": fstar,
                    "power_f": power_f,
                    "coherence": coherence,
                    "freqs": freqs,
                    "power": power,
                }

            W1 = analyze_window(eoi_arr, i_w1_start, i_w1_end)
            W2 = analyze_window(eoi_arr, i_w2_start, i_w2_end)
            W3 = analyze_window(eoi_arr, i_w3_start, i_w3_end)

            # persistence ratio: std(W3)/std(W2)
            def window_std(arr, i0, i1):
                if i0 is None or i1 is None or i1 - i0 < 1:
                    return 0.0
                return float(np.std(arr[i0 : i1 + 1]))

            std_W2 = window_std(eoi_arr, i_w2_start, i_w2_end)
            std_W3 = window_std(eoi_arr, i_w3_start, i_w3_end)
            persistence_ratio = (
                float(std_W3 / (std_W2 + 1e-12)) if std_W2 > 0 else float("nan")
            )

            # autocorr peak W3
            ac_peak_W3 = (
                autocorr_peak(eoi_arr[i_w3_start : i_w3_end + 1])
                if (i_w3_start is not None and i_w3_end >= i_w3_start)
                else 0.0
            )

            # phase-lock in W2 between EOI and entropy
            phase_lock = float("nan")
            if (
                i_w2_start is not None
                and i_w2_end is not None
                and i_w2_end - i_w2_start >= 4
            ):
                sub_eoi = eoi_arr[i_w2_start : i_w2_end + 1]
                sub_ent = np.array(entropy_logged[i_w2_start : i_w2_end + 1])
                if len(sub_eoi) == len(sub_ent) and len(sub_eoi) > 3:
                    xa_e = analytic_signal(sub_eoi)
                    xa_s = analytic_signal(sub_ent)
                    ph_e = np.angle(xa_e)
                    ph_s = np.angle(xa_s)
                    delta = ph_e - ph_s
                    phase_lock = float(np.mean(np.cos(delta)))

            # resync_time: compute phase before and after shock and time to return within 10 deg
            resync_time = float("nan")
            # compute analytic phase over full
            if len(eoi_arr) >= 8:
                xa_full = analytic_signal(eoi_arr)
                phase_full = np.angle(xa_full)
                # shock idx in logged
                shock_idx = epoch_index(t_shock)
                if shock_idx is not None and shock_idx < len(phase_full):
                    pre_start = max(0, shock_idx - (100 // LOG_EVERY))
                    pre_phase = phase_full[pre_start:shock_idx]
                    if pre_phase.size > 0:
                        mean_pre = np.angle(np.mean(np.exp(1j * pre_phase)))
                        for j in range(shock_idx, len(phase_full)):
                            cur = phase_full[j]
                            d = abs(
                                ((cur - mean_pre + math.pi) % (2 * math.pi)) - math.pi
                            )
                            if d <= math.radians(10.0):
                                resync_time = float(epochs_logged[j] - t_shock)
                                break

            mean_EOI = float(np.mean(eoi_arr)) if eoi_arr.size else 0.0
            mean_risk = float(np.mean(cr_norm)) if cr_norm.size else 0.0

            # pass criteria
            pass_flag = False
            try:
                if (
                    (W2["coherence"] >= 1.8)
                    and (not math.isnan(persistence_ratio) and persistence_ratio >= 0.6)
                    and (ac_peak_W3 >= 0.5)
                ):
                    pass_flag = True
            except Exception:
                pass_flag = False

            runs.append(
                {
                    "lambda": lam,
                    "amp": A,
                    "period": P,
                    "coupling": C,
                    "lag": dt_lag,
                    "seed": seed,
                    "mean_EOI": mean_EOI,
                    "mean_risk_norm": mean_risk,
                    "resync_time": resync_time,
                    "coherence_ratio_W2": W2["coherence"],
                    "persistence_ratio": persistence_ratio,
                    "autocorr_peak_W3": ac_peak_W3,
                    "phase_lock_W2": phase_lock,
                    "pass_flag": pass_flag,
                }
            )

            spectra.append(
                {
                    "lambda": lam,
                    "amp": A,
                    "period": P,
                    "coupling": C,
                    "lag": dt_lag,
                    "seed": seed,
                    "W1": W1,
                    "W2": W2,
                    "W3": W3,
                }
            )

            if pass_flag:
                candidates.append(
                    {
                        "lambda": lam,
                        "amp": A,
                        "period": P,
                        "coupling": C,
                        "lag": dt_lag,
                        "seed": seed,
                        "coherence_W2": W2["coherence"],
                        "persistence_ratio": persistence_ratio,
                        "autocorr_W3": ac_peak_W3,
                        "phase_lock": phase_lock,
                    }
                )

    # write outputs
    pd.DataFrame(runs).to_csv(DATA / "runs_phaseVI_summary.csv", index=False)
    # flatten spectra entries
    rows = []
    for s in spectra:
        rows.append(
            {
                "lambda": s["lambda"],
                "amp": s["amp"],
                "period": s["period"],
                "coupling": s["coupling"],
                "lag": s["lag"],
                "seed": s["seed"],
                "W1_fstar": s["W1"].get("fstar", float("nan")),
                "W1_power": s["W1"].get("power_f", float("nan")),
                "W1_coherence": s["W1"].get("coherence", float("nan")),
                "W2_fstar": s["W2"].get("fstar", float("nan")),
                "W2_power": s["W2"].get("power_f", float("nan")),
                "W2_coherence": s["W2"].get("coherence", float("nan")),
                "W3_fstar": s["W3"].get("fstar", float("nan")),
                "W3_power": s["W3"].get("power_f", float("nan")),
                "W3_coherence": s["W3"].get("coherence", float("nan")),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "freq_spectra_phaseVI.csv", index=False)
    pd.DataFrame(pulse_events).to_csv(DATA / "pulse_events_phaseVI.csv", index=False)

    if len(candidates) > 0:
        pd.DataFrame(candidates).to_csv(DATA / "heartbeat_candidates.csv", index=False)
    else:
        # produce near-miss top-3 by coherence_W2
        near = sorted(
            runs, key=lambda r: (r.get("coherence_ratio_W2") or 0.0), reverse=True
        )[:3]
        pd.DataFrame(near).to_csv(DATA / "near_miss_phaseVI.csv", index=False)

    # relaxation estimate: infer tau ~ best period from candidates or near-miss
    if len(candidates) > 0:
        best = sorted(candidates, key=lambda c: c["coherence_W2"], reverse=True)[0]
        tau_est = best["period"]
    else:
        tau_est = None
    with open(DATA / "relaxation_estimate.json", "w") as f:
        json.dump({"tau_estimate": tau_est}, f, indent=2)

    # plots: heatmap PASS density by period x amp per lambda (sampled grid may be sparse)
    df_runs = pd.DataFrame(runs)
    for lam in sorted(set(df_runs["lambda"].values)):
        sub = df_runs[df_runs["lambda"] == lam]
        if sub.empty:
            continue
        pivot = pd.pivot_table(
            sub,
            index="period",
            columns="amp",
            values="pass_flag",
            aggfunc="sum",
            fill_value=0,
        )
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="Greens")
        plt.colorbar(label="PASS count")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.title(f"PASS density period x amp (lambda={lam})")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "plots" / f"heatmap_pass_period_vs_amp_lambda_{lam}.png")
        plt.close()

    # lag sweep phase_lock (aggregate)
    if "lag" in df_runs.columns:
        plt.figure(figsize=(6, 4))
        for p in sorted(set(df_runs["period"].values)):
            sub = df_runs[df_runs["period"] == p]
            if sub.empty:
                continue
            grp = sub.groupby("lag")["phase_lock_W2"].mean()
            plt.plot(grp.index, grp.values, label=str(p))
        plt.xlabel("lag")
        plt.ylabel("phase_lock_W2")
        plt.legend(title="period")
        plt.title("lag sweep phase_lock")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "plots" / "lag_sweep_phase_lock.png")
        plt.close()

    # if candidates exist, pick best and plot EOI timeseries and spectra
    if len(candidates) > 0:
        best = candidates[0]
        # find matching entry in pulse_events to plot its EOI trace
        pedf = pd.DataFrame(pulse_events)
        match = pedf[
            (pedf["lambda"] == best["lambda"])
            & (pedf["amp"] == best["amp"])
            & (pedf["period"] == best["period"])
            & (pedf["coupling"] == best["coupling"])
            & (pedf["lag"] == best["lag"])
            & (pedf["seed"] == best["seed"])
        ]
        if not match.empty:
            # load eoi time series
            eoi_df = pd.DataFrame(pulse_events)
            sel = eoi_df[
                (eoi_df["lambda"] == best["lambda"])
                & (eoi_df["amp"] == best["amp"])
                & (eoi_df["period"] == best["period"])
                & (eoi_df["coupling"] == best["coupling"])
                & (eoi_df["lag"] == best["lag"])
                & (eoi_df["seed"] == best["seed"])
            ]
            if not sel.empty:
                grp = sel.groupby("epoch")["EOI"].mean()
                plt.figure(figsize=(8, 4))
                plt.plot(grp.index, grp.values)
                plt.axvspan(t_on, t_off, color="orange", alpha=0.1)
                plt.axvline(t_shock, color="red", linestyle="--")
                plt.title("EOI timeseries best candidate")
                plt.tight_layout()
                plt.savefig(OUT_DIR / "plots" / "eoi_timeseries_best.png")
                plt.close()

    # report md
    report_lines = []
    report_lines.append("# Phase VI — Resonant Universe")
    report_lines.append(f"Run stamp: {STAMP}")
    report_lines.append("")
    if len(candidates) > 0:
        report_lines.append("## HEARTBEAT CANDIDATES")
        best5 = sorted(candidates, key=lambda c: c["coherence_W2"], reverse=True)[:5]
        for c in best5:
            report_lines.append(
                f"- lambda={c['lambda']}, A={c['amp']}, P={c['period']}, C={c['coupling']}, lag={c['lag']}, seed={c['seed']}, coherence={c['coherence_W2']:.3f}, persistence={c['persistence_ratio']:.3f}, ac_W3={c['autocorr_W3']:.3f}, phase_lock={c['phase_lock']:.3f}"
            )
        report_lines.append("")
        report_lines.append(f"Estimated relaxation tau ≈ {tau_est}")
    else:
        report_lines.append("## NO RESONANCE FOUND")
        near = (
            pd.read_csv(DATA / "near_miss_phaseVI.csv")
            if (DATA / "near_miss_phaseVI.csv").exists()
            else pd.DataFrame(runs)
            .sort_values("coherence_ratio_W2", ascending=False)
            .head(3)
        )
        report_lines.append("Top near-miss configs:")
        for _, r in near.iterrows():
            report_lines.append(
                f"- lambda={r['lambda']}, A={r['amp']}, P={r['period']}, C={r['coupling']}, lag={r['lag']}, seed={r['seed']}, coherence={r.get('coherence_ratio_W2',0):.3f}"
            )

    with open(OUT_DIR / "panspermia_phaseVI_report.md", "w") as f:
        f.write("\n".join(report_lines))

    # bundle outputs
    bundle = OUT_DIR / f"panspermia_phaseVI_bundle_{STAMP}.zip"
    with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
        for f in (
            list(OUT_DIR.rglob("*"))
            + list(DATA.glob("*.csv"))
            + list(DATA.glob("*.json"))
        ):
            if f.is_file():
                z.write(f, arcname=str(f.relative_to(ROOT)))
    h = hashlib.sha256()
    with open(bundle, "rb") as bf:
        for chunk in iter(lambda: bf.read(1 << 20), b""):
            h.update(chunk)
    with open(OUT_DIR / "SHA256SUMS.txt", "w") as s:
        s.write(f"{h.hexdigest()}  {bundle.name}\n")

    # console takeaway
    lam_set = sorted(lam_grid)
    print("\nPHASE VI — RESONANT UNIVERSE")
    print(
        f"lambda in {lam_set}, amplitudes {amps}, periods {periods}, coupling={coupling_mode}, lags {lags}"
    )
    if len(candidates) > 0:
        best = sorted(candidates, key=lambda c: c["coherence_W2"], reverse=True)[0]
        print("Result: HEARTBEAT DETECTED ✅")
        print(
            f"Best params → lambda={best['lambda']}, A={best['amp']}, P={best['period']}, coupling={best['coupling']}, Δt={best['lag']}"
        )
        print(
            f"Evidence → coherence_W2={best['coherence_W2']:.3f}, persistence_ratio={best['persistence_ratio']:.3f}, autocorr_peak_W3={best['autocorr_W3']:.3f}, phase_lock_W2={best['phase_lock']:.3f}"
        )
        print(f"Estimated intrinsic relaxation time τ ≈ {tau_est}")
    else:
        print("Result: NO RESONANCE FOUND ❌")
        near = (
            pd.read_csv(DATA / "near_miss_phaseVI.csv")
            if (DATA / "near_miss_phaseVI.csv").exists()
            else pd.DataFrame(runs)
            .sort_values("coherence_ratio_W2", ascending=False)
            .head(3)
        )
        print("Top near-miss configs:")
        for _, r in near.iterrows():
            print(
                f"- lambda={r['lambda']}, A={r['amp']}, P={r['period']}, C={r['coupling']}, lag={r['lag']}, seed={r['seed']}, coherence={r.get('coherence_ratio_W2',0):.3f}"
            )
    print("\nOutputs:")
    print(str(bundle))


if __name__ == "__main__":
    main()
