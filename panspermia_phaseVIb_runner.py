#!/usr/bin/env python3
"""
Panspermia Phase VIb runner
Estimate relaxation time from closed run, then run a targeted sweep around τ with corrected normalization and robust spectral detection.
Defaults to sampled execution; use --full for exhaustive runs.
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

ROOT = Path(".")
DATA = ROOT / "data"
PLOTS = ROOT / "plots"
OUT = ROOT / "discovery_results"
for d in (DATA, PLOTS, OUT):
    d.mkdir(exist_ok=True)

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = OUT / f"panspermia_phaseVIb_{STAMP}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "plots").mkdir(exist_ok=True)


DEFAULTS = {
    "epochs": 14000,
    "seeds": [1, 2, 3, 4],
    "log_every": 10,
    "stability_window_last": 400,
    "base_epsilon": 0.0015,
    "lambda_grid": [1e-5, 7.5e-6, 5e-6, 3e-6, 2e-6],
    "amp_grid": [8, 12, 16, 24, 32],
    "width_ratio_grid": [0.06, 0.10, 0.15],
    "coupling_mode": "entropy_linked",
    "strength_grid": [0.2, 0.35, 0.5],
    "lag_grid": [0, 25, 50, 100, 200],
    "shock": {"enabled": True, "t": 7000, "intensity": 0.35, "duration": 25},
    "settle_phase": {"t_on": 2000, "t_off": 11000},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def ewma(arr, alpha):
    # simple EWMA
    s = []
    prev = None
    for x in arr:
        if prev is None:
            prev = x
        else:
            prev = alpha * x + (1 - alpha) * prev
        s.append(prev)
    return np.array(s)


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


def welch_psd(x, fs=1.0, nperseg=256, noverlap=None):
    # try scipy if available; otherwise implement simple Welch
    try:
        from scipy.signal import welch

        freqs, Pxx = welch(
            x, fs=fs, nperseg=nperseg, noverlap=noverlap or nperseg // 2, window="hann"
        )
        return freqs, Pxx
    except Exception:
        # simple periodogram with segmentation
        x = np.asarray(x)
        N = len(x)
        if N < 4:
            return np.array([]), np.array([])
        if nperseg is None or nperseg > N:
            nperseg = max(4, N // 4)
        step = nperseg - (noverlap or nperseg // 2)
        segments = []
        for start in range(0, N - nperseg + 1, step):
            seg = x[start : start + nperseg]
            w = np.hanning(nperseg)
            seg = seg * w
            X = np.fft.rfft(seg)
            P = (np.abs(X) ** 2) / (np.sum(w**2))
            segments.append(P)
        if not segments:
            return np.array([]), np.array([])
        Pxx = np.mean(np.vstack(segments), axis=0)
        freqs = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        return freqs, Pxx


def fisher_g_pvalue(periodogram, n_perm=200):
    # Monte Carlo p-value: shuffle data and recompute max-normalized power
    if len(periodogram) < 2:
        return 1.0
    g_obs = np.max(periodogram) / np.sum(periodogram)
    cnt = 0
    for _ in range(n_perm):
        perm = np.random.permutation(periodogram)
        g = np.max(perm) / np.sum(perm)
        if g >= g_obs:
            cnt += 1
    return float((cnt + 1) / (n_perm + 1))


def autocorr_peak(x):
    x = np.asarray(x)
    if x.size < 2:
        return 0.0
    x = x - x.mean()
    corr = np.correlate(x, x, mode="full")
    mid = len(corr) // 2
    ac = corr[mid + 1 :]
    ac = ac / ac[0] if ac[0] != 0 else ac
    return float(np.max(ac)) if ac.size > 0 else 0.0


def round_int_list(lst):
    return [int(round(x)) for x in lst]


def estimate_tau_from_closed(cfg):
    # run closed baseline with shocks disabled to estimate tau
    epochs = cfg["epochs"]
    seeds = cfg["seeds"]
    LOG_EVERY = cfg["log_every"]
    BASE_EPS = cfg["base_epsilon"]
    lam = 1e-5

    # single run aggregated across seeds (mean EOI)
    eoi_runs = []
    for seed in seeds:
        rng = np.random.RandomState(int(seed))
        agents = me.initialize_agents(200, 3, 0.05, rng)
        epochs_logged = []
        hazard_logged = []
        cci_logged = []
        eps_logged = []
        entropy_logged = []
        cci_max = 0.0
        for t in range(epochs):
            base = BASE_EPS * math.exp(-lam * t)
            eps_eff = base
            # no shock in closed run per spec
            me.step_update(agents, 0.0, "chronic", rng)
            hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(len(agents))
            cci = me.collective_cci(agents)
            cci_max = max(cci_max, cci)
            B = np.array([a["belief"] for a in agents]) if agents else np.array([])
            mean_b = B.mean(axis=0) if B.size else np.array([])
            S = -np.sum(mean_b * np.log(mean_b + 1e-12)) if mean_b.size else 0.0
            if t % LOG_EVERY == 0:
                epochs_logged.append(t)
                hazard_logged.append(hazard)
                cci_logged.append(cci)
                eps_logged.append(eps_eff)
                entropy_logged.append(S)
        # normalization via p1/p99 percentiles
        h_arr = np.array(hazard_logged)
        p1 = np.percentile(h_arr, 1)
        p99 = np.percentile(h_arr, 99)
        denom = p99 - p1 + 1e-9
        risk_norm = np.clip((h_arr - p1) / denom, 0.0, 1.0)
        risk_norm_s = ewma(risk_norm, 0.1)
        eps_ewma = ewma(np.array(eps_logged), 0.02)
        cci_arr = np.array(cci_logged)
        cci_scale = cci_arr / (max(1e-12, np.max(cci_arr)))
        eoi = eps_ewma * (1.0 - risk_norm_s) * cci_scale * 1.0 * 1.0
        eoi_runs.append(eoi)

    mean_eoi = np.mean(np.vstack(eoi_runs), axis=0)
    # detrend
    y = mean_eoi - np.mean(mean_eoi)
    # autocorrelation
    corr = np.correlate(y, y, mode="full")
    mid = len(corr) // 2
    ac = corr[mid:] / (corr[mid] if corr[mid] != 0 else 1.0)
    # find first lag where ac < exp(-1)
    thresh = math.exp(-1)
    tau = None
    for lag in range(1, len(ac)):
        if ac[lag] < thresh:
            tau = lag * cfg["log_every"]
            break
    if tau is None:
        tau = int(len(ac) * cfg["log_every"] / 4)

    return tau, mean_eoi, (epochs, epochs)


def main():
    args = parse_args()
    rnd = random.Random(args.seed)
    cfg = DEFAULTS

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

    print("Phase VIb anchors", astro_anchor, bio_anchor)

    # Step 2: Closed run estimate τ
    tau, mean_eoi, _ = estimate_tau_from_closed(cfg)
    print("Estimated tau (epochs):", tau)
    P_grid = round_int_list(
        [0.5 * tau, 0.75 * tau, 1.0 * tau, 1.25 * tau, 1.5 * tau, 2.0 * tau]
    )
    P_grid = sorted(set([p for p in P_grid if p > 0]))
    print("P_grid:", P_grid)
    # Save relaxation estimate
    with open(DATA / "relaxation_time.json", "w") as f:
        json.dump({"tau": tau, "P_grid": P_grid}, f, indent=2)

    # Build grid for driven runs
    lam_grid = cfg["lambda_grid"]
    amp_grid = cfg["amp_grid"]
    width_ratio_grid = cfg["width_ratio_grid"]
    strength_grid = cfg["strength_grid"]
    lag_grid = cfg["lag_grid"]
    seeds = cfg["seeds"]
    LOG_EVERY = cfg["log_every"]
    epochs = cfg["epochs"]
    t_on = cfg["settle_phase"]["t_on"]
    t_off = cfg["settle_phase"]["t_off"]
    shock_cfg = cfg["shock"]

    combos = []
    for lam in lam_grid:
        for P in P_grid:
            for A in amp_grid:
                for wr in width_ratio_grid:
                    for C in strength_grid:
                        for LAG in lag_grid:
                            combos.append((lam, P, A, wr, C, LAG))

    # sample if not full
    if not args.full:
        max_samples = 48
        if len(combos) > max_samples:
            combos = rnd.sample(combos, max_samples)
            print(
                f"[INFO] sampled {max_samples} combos out of {len(lam_grid)*len(P_grid)*len(amp_grid)*len(width_ratio_grid)*len(strength_grid)*len(lag_grid)}"
            )

    runs = []
    spectra = []
    candidates = []

    for lam, P, A, wr, C, LAG in combos:
        print("Running combo", lam, P, A, wr, C, LAG)
        width = max(1, int(wr * P))
        pulse_centers = list(range(t_on, t_off, P))
        for seed in seeds:
            rng = np.random.RandomState(int(seed + (int(lam * 1e9) % 100000)))
            agents = me.initialize_agents(200, 3, 0.05, rng)
            epochs_logged = []
            hazard_logged = []
            cci_logged = []
            eps_logged = []
            entropy_logged = []
            cci_max = 0.0
            S_hist = []
            CCI_hist = []
            for t in range(epochs):
                base = cfg["base_epsilon"] * math.exp(-lam * t)
                eps_eff = base
                if t_on <= t < t_off:
                    env = 0.0
                    for center in pulse_centers:
                        dtc = t - center
                        if abs(dtc) > 4 * width:
                            continue
                        env += math.exp(-((dtc) ** 2) / (2 * (width / 2.0) ** 2))
                    if env > 0:
                        eps_pulse = base * (1.0 + A * env)
                    else:
                        eps_pulse = base
                    # feedback
                    f_mult = 1.0
                    t_lag = t - LAG
                    if cfg["coupling_mode"] == "entropy_linked":
                        if t_lag >= 0 and t_lag < len(S_hist):
                            S_lag = S_hist[t_lag]
                            S_max = math.log(max(2, len(agents)))
                            f_mult = 1.0 + C * (1.0 - (S_lag / (S_max + 1e-12)))
                    else:
                        if t_lag >= 0 and t_lag < len(CCI_hist):
                            cci_lag = CCI_hist[t_lag]
                            f_mult = (
                                1.0 + C * (cci_lag / (cci_max + 1e-12))
                                if cci_max > 0
                                else 1.0
                            )
                    eps_eff = eps_pulse * f_mult
                # shock
                if (
                    shock_cfg["enabled"]
                    and shock_cfg["t"] <= t < shock_cfg["t"] + shock_cfg["duration"]
                ):
                    current_shock = shock_cfg["intensity"]
                    for a in agents:
                        if a["alive"]:
                            a["resource"] -= current_shock * 0.2
                            if a["resource"] < 0:
                                a["alive"] = False
                else:
                    current_shock = 0.0
                if eps_eff > 0:
                    for a in [a for a in agents if a["alive"]]:
                        a["resource"] = min(1.0, a["resource"] + eps_eff)
                me.step_update(agents, current_shock, "chronic", rng)
                hazard = sum(1 for a in agents if a["resource"] < 0.2) / float(
                    len(agents)
                )
                cci = me.collective_cci(agents)
                cci_max = max(cci_max, cci)
                B = np.array([a["belief"] for a in agents]) if agents else np.array([])
                mean_b = B.mean(axis=0) if B.size else np.array([])
                S = -np.sum(mean_b * np.log(mean_b + 1e-12)) if mean_b.size else 0.0
                S_hist.append(S)
                CCI_hist.append(cci)
                if t % LOG_EVERY == 0:
                    epochs_logged.append(t)
                    hazard_logged.append(hazard)
                    cci_logged.append(cci)
                    eps_logged.append(eps_eff)
                    entropy_logged.append(S)

            # normalize risk using p1/p99 (robust) and EWMA
            h_arr = np.array(hazard_logged)
            p1 = np.percentile(h_arr, 1)
            p99 = np.percentile(h_arr, 99)
            denom = p99 - p1 + 1e-9
            risk_norm = np.clip((h_arr - p1) / denom, 0.0, 1.0)
            risk_norm_s = ewma(risk_norm, 0.1)
            eps_ewma = ewma(np.array(eps_logged), 0.02)
            cci_arr = np.array(cci_logged)
            cci_scale = cci_arr / (max(1e-12, np.max(cci_arr)))
            eoi = eps_ewma * (1.0 - risk_norm_s) * cci_scale * astro_anchor * bio_anchor
            eoi = np.clip(eoi, 0.0, 1.0)

            # spectral windows (indices)
            def idx_for(e):
                arr = np.array(epochs_logged)
                if arr.size == 0:
                    return None
                i = np.searchsorted(arr, e)
                if i >= len(arr):
                    return len(arr) - 1
                return i

            i_w1_s = idx_for(t_on)
            i_w1_e = idx_for(shock_cfg["t"])
            i_w2_s = idx_for(shock_cfg["t"] + shock_cfg["duration"])
            i_w2_e = idx_for(t_off)
            i_w3_s = idx_for(t_off)
            i_w3_e = len(eoi) - 1

            def analyze_win(arr, i0, i1):
                if i0 is None or i1 is None or i1 - i0 < 4:
                    return {
                        "fstar": float("nan"),
                        "power_f": float("nan"),
                        "coherence": float("nan"),
                        "p_g": 1.0,
                    }
                sub = arr[i0 : i1 + 1]
                sub = sub - np.mean(sub)
                # apply hann and welch
                freqs, Pxx = welch_psd(
                    sub, fs=1.0 / (LOG_EVERY), nperseg=min(256, len(sub))
                )
                if freqs.size <= 1 or np.sum(Pxx) == 0:
                    return {
                        "fstar": float("nan"),
                        "power_f": float("nan"),
                        "coherence": float("nan"),
                        "p_g": 1.0,
                    }
                # exclude DC (freq 0)
                idx = np.argmax(Pxx[1:]) + 1
                fstar = freqs[idx]
                power_f = float(Pxx[idx])
                power_bg = float(np.median(np.delete(Pxx, idx)))
                coherence = float(power_f / (power_bg + 1e-12))
                # fisher g p-value via Monte Carlo on periodogram
                p_g = fisher_g_pvalue(Pxx, n_perm=200)
                return {
                    "fstar": fstar,
                    "power_f": power_f,
                    "coherence": coherence,
                    "p_g": p_g,
                    "freqs": freqs,
                    "Pxx": Pxx,
                }

            W1 = analyze_win(eoi, i_w1_s, i_w1_e)
            W2 = analyze_win(eoi, i_w2_s, i_w2_e)
            W3 = analyze_win(eoi, i_w3_s, i_w3_e)

            std_W2 = (
                np.std(eoi[i_w2_s : i_w2_e + 1])
                if (i_w2_s is not None and i_w2_e is not None and i_w2_e > i_w2_s)
                else 0.0
            )
            std_W3 = (
                np.std(eoi[i_w3_s : i_w3_e + 1])
                if (i_w3_s is not None and i_w3_e is not None and i_w3_e > i_w3_s)
                else 0.0
            )
            persistence_ratio = (
                float(std_W3 / (std_W2 + 1e-12)) if std_W2 > 0 else float("nan")
            )
            ac_peak_W3 = (
                autocorr_peak(eoi[i_w3_s : i_w3_e + 1]) if i_w3_s is not None else 0.0
            )

            # phase-lock in W2 (EOI vs entropy)
            phase_lock = float("nan")
            if i_w2_s is not None and i_w2_e is not None and i_w2_e - i_w2_s >= 4:
                sub_eoi = eoi[i_w2_s : i_w2_e + 1]
                sub_ent = np.array(entropy_logged[i_w2_s : i_w2_e + 1])
                if len(sub_eoi) == len(sub_ent) and len(sub_eoi) > 3:
                    ph_e = np.angle(analytic_signal(sub_eoi))
                    ph_s = np.angle(analytic_signal(sub_ent))
                    delta = ph_e - ph_s
                    phase_lock = float(np.abs(np.mean(np.exp(1j * delta))))

            # pass criteria
            pass_flag = False
            try:
                if (
                    (W2["coherence"] >= 2.0)
                    and (W2["p_g"] < 0.05)
                    and (ac_peak_W3 >= 0.55)
                    and (
                        not math.isnan(persistence_ratio) and persistence_ratio >= 0.65
                    )
                ):
                    pass_flag = True
            except Exception:
                pass_flag = False

            runs.append(
                {
                    "lambda": lam,
                    "period": P,
                    "amp": A,
                    "width_ratio": wr,
                    "coupling": cfg["coupling_mode"],
                    "strength": C,
                    "lag": LAG,
                    "seed": seed,
                    "mean_EOI": float(np.mean(eoi)),
                    "mean_risk_norm": float(np.mean(risk_norm_s)),
                    "resync_time": float("nan"),
                    "coherence_ratio_W2": W2["coherence"],
                    "p_g_W2": W2["p_g"],
                    "persistence_ratio": persistence_ratio,
                    "autocorr_peak_W3": ac_peak_W3,
                    "phase_lock_W2": phase_lock,
                    "pass_flag": pass_flag,
                }
            )

            spectra.append(
                {
                    "lambda": lam,
                    "period": P,
                    "amp": A,
                    "width_ratio": wr,
                    "strength": C,
                    "lag": LAG,
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
                        "period": P,
                        "amp": A,
                        "width_ratio": wr,
                        "strength": C,
                        "lag": LAG,
                        "seed": seed,
                        "coherence_W2": W2["coherence"],
                        "p_g": W2["p_g"],
                        "persistence_ratio": persistence_ratio,
                        "ac_W3": ac_peak_W3,
                        "phase_lock": phase_lock,
                    }
                )

    # write outputs
    pd.DataFrame(runs).to_csv(DATA / "runs_phaseVIb_summary.csv", index=False)
    rows = []
    for s in spectra:
        rows.append(
            {
                "lambda": s["lambda"],
                "period": s["period"],
                "amp": s["amp"],
                "width_ratio": s["width_ratio"],
                "strength": s["strength"],
                "lag": s["lag"],
                "seed": s["seed"],
                "W1_fstar": s["W1"].get("fstar", float("nan")),
                "W1_power": s["W1"].get("power_f", float("nan")),
                "W1_coh": s["W1"].get("coherence", float("nan")),
                "W1_p": s["W1"].get("p_g", 1.0),
                "W2_fstar": s["W2"].get("fstar", float("nan")),
                "W2_power": s["W2"].get("power_f", float("nan")),
                "W2_coh": s["W2"].get("coherence", float("nan")),
                "W2_p": s["W2"].get("p_g", 1.0),
                "W3_fstar": s["W3"].get("fstar", float("nan")),
                "W3_power": s["W3"].get("power_f", float("nan")),
                "W3_coh": s["W3"].get("coherence", float("nan")),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "freq_spectra_phaseVIb.csv", index=False)

    if len(candidates) > 0:
        pd.DataFrame(candidates).to_csv(
            DATA / "heartbeat_candidates_phaseVIb.csv", index=False
        )
    else:
        # write near-miss top-3
        near = sorted(
            runs, key=lambda r: (r.get("coherence_ratio_W2") or 0.0), reverse=True
        )[:3]
        pd.DataFrame(near).to_csv(DATA / "near_miss_phaseVIb.csv", index=False)

    with open(DATA / "relaxation_time.json", "w") as f:
        json.dump({"tau": tau, "P_grid": P_grid}, f, indent=2)

    # plots: heatmap pass vs period/amp per lambda
    df = pd.DataFrame(runs)
    for lam in sorted(set(df["lambda"].values)):
        sub = df[df["lambda"] == lam]
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
        plt.savefig(OUT_DIR / "plots" / f"heatmap_pass_vs_period_amp_lambda_{lam}.png")
        plt.close()

    # lag sweep
    if "lag" in df.columns:
        plt.figure(figsize=(6, 4))
        for p in sorted(set(df["period"].values)):
            sub = df[df["period"] == p]
            if sub.empty:
                continue
            grp = sub.groupby("lag")["phase_lock_W2"].mean()
            plt.plot(grp.index, grp.values, label=str(p))
        plt.xlabel("lag")
        plt.ylabel("phase_lock_W2")
        plt.legend(title="period")
        plt.title("lag sweep phase_lock")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "plots" / "lag_sweep_phase_lock_VIb.png")
        plt.close()

    # best candidate plots
    if len(candidates) > 0:
        best = sorted(candidates, key=lambda c: c["coherence_W2"], reverse=True)[0]
        # plot its EOI timeseries from data file
        pedf = (
            pd.DataFrame(pd.read_csv(DATA / "pulse_events_phaseVI.csv"))
            if (DATA / "pulse_events_phaseVI.csv").exists()
            else None
        )
        # save report
        lines = ["# Phase VIb — Resonance re-test", "", f"Tau estimate: {tau}", ""]
        lines.append("Top candidates:")
        for c in sorted(candidates, key=lambda c: c["coherence_W2"], reverse=True)[:5]:
            lines.append(str(c))
        with open(OUT_DIR / "panspermia_phaseVIb_report.md", "w") as f:
            f.write("\n".join(lines))
    else:
        lines = [
            "# Phase VIb — Resonance re-test",
            "",
            f"Tau estimate: {tau}",
            "",
            "NO RESONANCE FOUND",
            "Top near-miss:",
        ]
        near = (
            pd.read_csv(DATA / "near_miss_phaseVIb.csv")
            if (DATA / "near_miss_phaseVIb.csv").exists()
            else pd.DataFrame(runs)
            .sort_values("coherence_ratio_W2", ascending=False)
            .head(3)
        )
        for _, r in near.iterrows():
            lines.append(str(r.to_dict()))
        with open(OUT_DIR / "panspermia_phaseVIb_report.md", "w") as f:
            f.write("\n".join(lines))

    # bundle
    bundle = OUT_DIR / f"panspermia_phaseVIb_bundle_{STAMP}.zip"
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

    # console summary
    print("\nPHASE VIb — RESONANCE RE-TEST")
    print(f"Tau ≈ {tau} epochs; P_grid={P_grid}")
    if len(candidates) > 0:
        print("Result: HEARTBEAT DETECTED ✅")
        best = sorted(candidates, key=lambda c: c["coherence_W2"], reverse=True)[0]
        print("Best params:", best)
    else:
        print("Result: NO RESONANCE FOUND ❌")
        near = (
            pd.read_csv(DATA / "near_miss_phaseVIb.csv")
            if (DATA / "near_miss_phaseVIb.csv").exists()
            else pd.DataFrame(runs)
            .sort_values("coherence_ratio_W2", ascending=False)
            .head(3)
        )
        print("Top near-miss:")
        for _, r in near.iterrows():
            print(r.to_dict())
    print("Outputs:", str(bundle))


if __name__ == "__main__":
    main()
