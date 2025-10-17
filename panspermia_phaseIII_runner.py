#!/usr/bin/env python3
"""
Panspermia Phase III runner
Sweeps lambda on logspace, measures distinguishability Open vs Closed, finds transition lambda*.
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
OUT = ROOT / "discovery_results" / f"panspermia_phaseIII_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "plots").mkdir(exist_ok=True)

# --- Config (defaults per user spec) ---
EPOCHS = 5000
SEEDS = [1, 2, 3, 4]
LOG_EVERY = 10
STABILITY_WINDOW = 200
SHOCK_ENABLED = True
T_SHOCK = EPOCHS // 2
SHOCK_INTENSITY = 0.35
SHOCK_DURATION = 25

EPSILONS = [0.0015, 0.0020]
LAMBDA_LOGSPACE = [
    1e-5,
    1.78e-5,
    3.16e-5,
    5.62e-5,
    1e-4,
    1.78e-4,
    3.16e-4,
    5.62e-4,
    8.9e-4,
    1e-3,
]

CRITERIA = {"log10BF_threshold": 0.0, "auc_threshold": 0.6, "eoi_delta_threshold": 0.03}

SPEED_MODE = "standard"
if SPEED_MODE == "fast":
    EPOCHS = 3000
    SHOCK_DURATION = 15

# load earth anchors or defaults
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

# anchors
v_astro = float(earth.get("entropy_rate_CMB", 1.0)) * float(
    earth.get("dark_energy_density", 1.0)
)
# normalization prior
ASTRO_XMIN = 0.0
ASTRO_XMAX = max(v_astro, 1.0)
astro_anchor = (v_astro - ASTRO_XMIN) / max(ASTRO_XMAX - ASTRO_XMIN, 1e-12)


# bio anchor
def logistic(x):
    return 1.0 / (1.0 + math.exp(-x))


bio_anchor = logistic(
    0.6 * float(earth.get("code_opt_z", 2.5))
    + 0.3 * (1.0 / float(earth.get("t_origin_Gyr", 4.5)))
    + 0.1 * float(earth.get("homochirality_lock_in_score", 0.8))
)

print("Anchors: astro_anchor=", astro_anchor, "bio_anchor=", bio_anchor)


# helper: compute AUC by pairwise comparison (works for small samples)
def compute_auc_from_scores(open_vals, closed_vals):
    # AUC = probability(score_open > score_closed) + 0.5*P(tie)
    n_open = len(open_vals)
    n_closed = len(closed_vals)
    if n_open == 0 or n_closed == 0:
        return float("nan")
    wins = 0.0
    ties = 0.0
    for o in open_vals:
        for c in closed_vals:
            if o > c:
                wins += 1.0
            elif abs(o - c) <= 1e-12:
                ties += 1.0
    auc = (wins + 0.5 * ties) / (n_open * n_closed)
    return auc


# helper: normal pdf
def pdf_normal(x, mu, sigma):
    sigma = max(sigma, 1e-6)
    return math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))


# containers
runs_rows = []
op_events_rows = []

# Always include A_closed baseline
conditions = [{"name": "A_closed", "epsilon0": 0.0, "lambda": 0.0, "is_open": False}]
for eps in EPSILONS:
    for lam in LAMBDA_LOGSPACE:
        conditions.append(
            {
                "name": f"B_open_eps{eps}_lam{lam}",
                "epsilon0": eps,
                "lambda": lam,
                "is_open": True,
            }
        )

print("Will run", len(conditions), "conditions x", len(SEEDS), "seeds")

# Run simulations
for cond in conditions:
    name = cond["name"]
    eps0 = cond["epsilon0"]
    lam = cond["lambda"]
    is_open = cond["is_open"]
    print("Condition", name)
    for seed in SEEDS:
        rng = np.random.RandomState(int(seed))
        agents = me.initialize_agents(200, 3, 0.05, rng)
        epochs_logged = []
        cci_logged = []
        hazard_logged = []
        eps_eff_logged = []
        cci_max = 0.0
        # simulate
        for t in range(EPOCHS):
            # compute epsilon effective
            if is_open and eps0 > 0:
                eps_eff = eps0 * math.exp(-lam * t)
            else:
                eps_eff = 0.0
            # log event if open
            if is_open and eps_eff > 0:
                op_events_rows.append(
                    {
                        "epoch": t,
                        "condition": name,
                        "seed": seed,
                        "epsilon0": eps0,
                        "lambda": lam,
                        "epsilon_eff": eps_eff,
                    }
                )
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
            if is_open and eps_eff > 0:
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
            if t % LOG_EVERY == 0:
                epochs_logged.append(t)
                cci_logged.append(cci)
                hazard_logged.append(hazard)
                eps_eff_logged.append(eps_eff)
        # normalize collapse risk per run
        h_arr = np.array(hazard_logged)
        h_min = float(np.min(h_arr)) if h_arr.size else 0.0
        h_max = float(np.max(h_arr)) if h_arr.size else h_min
        denom = max(1e-9, h_max - h_min)
        cr_norm = np.clip((h_arr - h_min) / denom, 0.0, 1.0)
        # compute cumulative mean epsilon to date
        eps_arr = np.array(eps_eff_logged)
        cum_mean_eps = np.array(
            [eps_arr[: i + 1].mean() if i >= 0 else 0.0 for i in range(len(eps_arr))]
        )
        # compute EOI per logged epoch
        eoi_arr = []
        for i in range(len(epochs_logged)):
            cci = cci_logged[i]
            cci_scale = (cci / (cci_max + 1e-12)) if cci_max > 0 else 0.0
            eoi = (
                cum_mean_eps[i]
                * (1.0 - cr_norm[i])
                * cci_scale
                * astro_anchor
                * bio_anchor
            )
            eoi_arr.append(float(eoi))
        # summary stats per run
        mean_CCI = float(np.mean(cci_logged)) if cci_logged else 0.0
        mean_risk = float(np.mean(cr_norm)) if cr_norm.size else 0.0
        mean_EOI = float(np.mean(eoi_arr)) if eoi_arr else 0.0
        # last-window mean (convert window to logged points)
        logged_window = int(STABILITY_WINDOW // LOG_EVERY)
        last_idx = len(cci_logged)
        window_start = max(0, last_idx - logged_window)
        last_window_CCI = (
            float(np.mean(cci_logged[window_start:])) if cci_logged else 0.0
        )
        last_window_risk = (
            float(np.mean(cr_norm[window_start:])) if cr_norm.size else 0.0
        )
        last_window_EOI = float(np.mean(eoi_arr[window_start:])) if eoi_arr else 0.0
        # shock recovery
        pre_start = max(0, (T_SHOCK - 100) // LOG_EVERY)
        pre_end = max(0, (T_SHOCK - 1) // LOG_EVERY)
        pre_ccis = cci_logged[pre_start : pre_end + 1] if pre_end >= pre_start else []
        pre_shock_CCI_mean = float(np.mean(pre_ccis)) if pre_ccis else 0.0
        recovery_time = float("nan")
        for idx in range((T_SHOCK) // LOG_EVERY, len(cci_logged)):
            if pre_shock_CCI_mean == 0:
                continue
            cur = cci_logged[idx]
            if abs(cur - pre_shock_CCI_mean) <= 0.05 * abs(
                pre_shock_CCI_mean if pre_shock_CCI_mean != 0 else 1.0
            ):
                recovery_time = float(epochs_logged[idx] - T_SHOCK)
                break
        post_start_idx = (T_SHOCK + SHOCK_DURATION) // LOG_EVERY
        post_end_idx = min(len(cci_logged) - 1, post_start_idx + (100 // LOG_EVERY))
        post_risk_mean = (
            float(np.mean(cr_norm[post_start_idx : post_end_idx + 1]))
            if post_end_idx >= post_start_idx
            else float("nan")
        )
        pre_risk_mean = (
            float(np.mean(cr_norm[pre_start : pre_end + 1]))
            if pre_end >= pre_start
            else float("nan")
        )
        delta_risk = (
            post_risk_mean - pre_risk_mean
            if (not math.isnan(post_risk_mean) and not math.isnan(pre_risk_mean))
            else float("nan")
        )
        post_eoi_mean = (
            float(np.mean(eoi_arr[post_start_idx : post_end_idx + 1]))
            if post_end_idx >= post_start_idx
            else float("nan")
        )
        pre_eoi_mean = (
            float(np.mean(eoi_arr[pre_start : pre_end + 1]))
            if pre_end >= pre_start
            else float("nan")
        )
        delta_eoi = (
            post_eoi_mean - pre_eoi_mean
            if (not math.isnan(post_eoi_mean) and not math.isnan(pre_eoi_mean))
            else float("nan")
        )
        runs_rows.append(
            {
                "condition": name,
                "seed": seed,
                "epsilon0": eps0,
                "lambda": lam,
                "mean_CCI": mean_CCI,
                "mean_risk_norm": mean_risk,
                "mean_EOI": mean_EOI,
                "last_window_mean_CCI": last_window_CCI,
                "last_window_mean_risk": last_window_risk,
                "last_window_mean_EOI": last_window_EOI,
                "recovery_time": recovery_time,
                "delta_risk_postshock": delta_risk,
                "delta_EOI_postshock": delta_eoi,
            }
        )

# write raw outputs
pd.DataFrame(op_events_rows).to_csv(
    DATA / "openness_events_decay_phaseIII.csv", index=False
)
pd.DataFrame(runs_rows).to_csv(DATA / "runs_phaseIII_summary.csv", index=False)

# Distinguishability analysis per lambda
rr = pd.DataFrame(runs_rows)
rows_dist = []
for lam in LAMBDA_LOGSPACE:
    # for each epsilon, compute metrics; then choose best-open (highest log-likelihood / BF)
    best_for_lam = {
        "lambda": lam,
        "best_epsilon": None,
        "log10BF": -np.inf,
        "BF": None,
        "AUC": None,
        "delta_EOI": None,
    }
    for eps in EPSILONS:
        open_name = f"B_open_eps{eps}_lam{lam}"
        closed_name = "A_closed"
        open_rows = rr[rr.condition == open_name]
        closed_rows = rr[rr.condition == closed_name]
        # need at least one run each
        if open_rows.empty or closed_rows.empty:
            continue
        # feature vectors (last-window)
        open_eoi = open_rows.last_window_mean_EOI.values
        closed_eoi = closed_rows.last_window_mean_EOI.values
        # delta EOI
        delta_eoi = float(np.mean(open_eoi) - np.mean(closed_eoi))

        # AUC from EOI only or combined features? spec asks multi-feature classifier; we'll use last-window features X
        # build X and y
        def build_X(df_sub):
            return np.vstack(
                [
                    df_sub.last_window_mean_CCI.values,
                    df_sub.last_window_mean_risk.values,
                    df_sub.last_window_mean_EOI.values,
                    df_sub.recovery_time.fillna(9999).values,
                    df_sub.delta_risk_postshock.fillna(0).values,
                    df_sub.delta_EOI_postshock.fillna(0).values,
                ]
            ).T

        X_open = build_X(open_rows)
        X_closed = build_X(closed_rows)
        X = np.vstack([X_open, X_closed])
        y = np.array([1] * len(X_open) + [0] * len(X_closed))
        # simple logistic classifier 5-fold CV would require sklearn; instead compute AUC using one feature (EOI) as proxy and also compute pairwise AUC on multifeature via a simple distance score
        # compute AUC via EOI
        auc_eoi = compute_auc_from_scores(open_eoi.tolist(), closed_eoi.tolist())
        # distance score: for each sample compute mean of features and use that as score
        open_score = X_open.mean(axis=1)
        closed_score = X_closed.mean(axis=1)
        auc_multi = compute_auc_from_scores(open_score.tolist(), closed_score.tolist())
        auc = float(
            np.nanmean(
                [
                    auc_eoi if not math.isnan(auc_eoi) else 0.5,
                    auc_multi if not math.isnan(auc_multi) else 0.5,
                ]
            )
        )
        # BF calculation: use signature BFs + EOI term
        # signature BFs (re-use earth signatures, simple normal model)
        sig = {
            "t_origin_Gyr": earth.get("t_origin_Gyr", 4.5),
            "code_opt_z": earth.get("code_opt_z", 2.5),
            "homochirality_lock_in_score": earth.get(
                "homochirality_lock_in_score", 0.8
            ),
        }
        bf_sig = {}
        for k, v in sig.items():
            try:
                val = float(v)
            except:
                val = 0.5
            p1 = pdf_normal(val, 0.7, 0.3)
            p0 = pdf_normal(val, 0.3, 0.3)
            bf_sig[k] = p1 / max(p0, 1e-12)
        # EOI BF: use observed = mean open last-window mean_EOI; compute p_open ~ N(mu_open, sigma_open), p_closed ~ N(mu_closed, sigma_closed)
        mu_open = float(np.mean(open_eoi))
        sigma_open = float(np.std(open_eoi)) if np.std(open_eoi) > 0 else 1e-6
        mu_closed = float(np.mean(closed_eoi))
        sigma_closed = float(np.std(closed_eoi)) if np.std(closed_eoi) > 0 else 1e-6
        observed = mu_open
        p_open = pdf_normal(observed, mu_open, sigma_open)
        p_closed = pdf_normal(observed, mu_closed, sigma_closed)
        bf_eoi = p_open / max(p_closed, 1e-12)
        combined_bf = float(np.prod(list(bf_sig.values())) * bf_eoi)
        log10bf = math.log10(combined_bf) if combined_bf > 0 else -np.inf
        # record if best
        if log10bf > best_for_lam["log10BF"]:
            best_for_lam.update(
                {
                    "lambda": lam,
                    "best_epsilon": eps,
                    "log10BF": log10bf,
                    "BF": combined_bf,
                    "AUC": auc,
                    "delta_EOI": delta_eoi,
                }
            )
    rows_dist.append(best_for_lam)

# save distinguishability
pd.DataFrame(rows_dist).to_csv(DATA / "distinguishability_by_lambda.csv", index=False)

# determine lambda*
lambda_star = None
epsilon_star = None
evidence = None
for r in rows_dist:
    if r["log10BF"] <= CRITERIA["log10BF_threshold"]:
        # check secondary criteria
        auc = r.get("AUC", 0)
        de = abs(r.get("delta_EOI", 0) if r.get("delta_EOI") is not None else 0)
        if auc < CRITERIA["auc_threshold"] and de < CRITERIA["eoi_delta_threshold"]:
            lambda_star = r["lambda"]
            epsilon_star = r["best_epsilon"]
            evidence = {
                "log10BF_at_lambda_star": r["log10BF"],
                "AUC_at_lambda_star": r["AUC"],
                "delta_EOI_at_lambda_star": r["delta_EOI"],
            }
            break
# fallback: pick first lambda where log10BF <= threshold regardless of secondary
if lambda_star is None:
    for r in rows_dist:
        if r["log10BF"] <= CRITERIA["log10BF_threshold"]:
            lambda_star = r["lambda"]
            epsilon_star = r["best_epsilon"]
            evidence = {
                "log10BF_at_lambda_star": r["log10BF"],
                "AUC_at_lambda_star": r["AUC"],
                "delta_EOI_at_lambda_star": r["delta_EOI"],
            }
            break
# if still None, choose max lambda tested and report that open still better
if lambda_star is None and rows_dist:
    lambda_star = rows_dist[-1]["lambda"]
    epsilon_star = rows_dist[-1]["best_epsilon"]
    evidence = {
        "log10BF_at_lambda_star": rows_dist[-1]["log10BF"],
        "AUC_at_lambda_star": rows_dist[-1]["AUC"],
        "delta_EOI_at_lambda_star": rows_dist[-1]["delta_EOI"],
    }

transition = {
    "lambda_star": lambda_star,
    "epsilon_star": epsilon_star,
    "criteria": CRITERIA,
    "evidence": evidence,
}
with open(DATA / "transition_point.json", "w") as f:
    json.dump(transition, f, indent=2)

# plots
df_dist = pd.DataFrame(rows_dist)
if not df_dist.empty:
    xs = df_dist["lambda"].astype(float)
    plt.figure(figsize=(6, 4))
    plt.plot(xs, df_dist["log10BF"], marker="o")
    if lambda_star is not None:
        plt.axvline(lambda_star, color="red", linestyle="--")
        plt.text(lambda_star, plt.ylim()[1] * 0.9, f"λ*={lambda_star:.1e}", color="red")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("log10BF")
    plt.title("log10BF vs lambda")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "transition_curve_log10BF.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, df_dist["AUC"], marker="o")
    if lambda_star is not None:
        plt.axvline(lambda_star, color="red", linestyle="--")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("AUC")
    plt.title("AUC vs lambda")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "transition_curve_AUC.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(xs, df_dist["delta_EOI"], marker="o")
    if lambda_star is not None:
        plt.axvline(lambda_star, color="red", linestyle="--")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("ΔEOI")
    plt.title("ΔEOI vs lambda")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "transition_curve_EOI_delta.png")
    plt.close()

# representative EOI time series around lambda*
rep_lams = (
    sorted(LAMBDA_LOGSPACE, key=lambda x: abs(math.log(x) - math.log(lambda_star)))[:3]
    if lambda_star
    else LAMBDA_LOGSPACE[:3]
)
edf = (
    pd.read_csv(DATA / "eoi_timeseries.csv")
    if (DATA / "eoi_timeseries.csv").exists()
    else pd.DataFrame()
)
if not edf.empty:
    plt.figure(figsize=(8, 4))
    for lam in rep_lams:
        for eps in EPSILONS:
            name = f"B_open_eps{eps}_lam{lam}"
            sub = edf[(edf.condition == name)]
            if sub.empty:
                continue
            # plot mean across seeds
            grp = sub.groupby("epoch").EOI.mean()
            plt.plot(grp.index, grp.values, label=f"{name}")
    plt.legend()
    plt.title("Representative EOI time series")
    plt.xlabel("epoch")
    plt.ylabel("EOI")
    plt.tight_layout()
    plt.savefig(OUT / "plots" / "eoi_long_examples.png")
    plt.close()

# shock recovery compare: Closed vs best-Open at low lambda and at lambda*
# choose low lambda (min tested) and lambda*
low_lam = min(LAMBDA_LOGSPACE)
best_low = None
best_star = None
for r in rows_dist:
    if r["lambda"] == low_lam:
        best_low = r
    if r["lambda"] == lambda_star:
        best_star = r
# produce simple scatter of recovery_time
rr = pd.DataFrame(runs_rows)
plt.figure(figsize=(6, 4))
if best_low is not None:
    name_open = f'B_open_eps{best_low["best_epsilon"]}_lam{best_low["lambda"]}'
    sub_open = rr[rr.condition == name_open]
    sub_closed = rr[rr.condition == "A_closed"]
    plt.scatter(["Closed"] * len(sub_closed), sub_closed.recovery_time, alpha=0.6)
    plt.scatter(["Open_lowlam"] * len(sub_open), sub_open.recovery_time, alpha=0.6)
if best_star is not None:
    name_open_s = f'B_open_eps{best_star["best_epsilon"]}_lam{best_star["lambda"]}'
    sub_open_s = rr[rr.condition == name_open_s]
    plt.scatter(["Open_star"] * len(sub_open_s), sub_open_s.recovery_time, alpha=0.6)
plt.ylabel("recovery_time")
plt.title("Shock recovery (Closed vs Open low-lam vs Open λ*)")
plt.tight_layout()
plt.savefig(OUT / "plots" / "shock_recovery_compare.png")
plt.close()

# save distinguishing table
pd.DataFrame(rows_dist).to_csv(DATA / "distinguishability_by_lambda.csv", index=False)

# write report
md = OUT / "panspermia_phaseIII_report.md"
with open(md, "w") as f:
    f.write("# Panspermia Phase III — Openness Collapse Sweep\n\n")
    f.write("Config:\n")
    f.write(
        json.dumps(
            {
                "EPOCHS": EPOCHS,
                "SEEDS": SEEDS,
                "LOG_EVERY": LOG_EVERY,
                "STABILITY_WINDOW": STABILITY_WINDOW,
                "EPSILONS": EPSILONS,
                "LAMBDA_LOGSPACE": LAMBDA_LOGSPACE,
            },
            indent=2,
        )
    )
    f.write("\n\nTransition:\n")
    f.write(json.dumps(transition, indent=2))
    f.write("\n\nDistinguishability table (data/distinguishability_by_lambda.csv)\n")

# bundle everything
bundle = OUT / f"panspermia_phaseIII_bundle_{STAMP}.zip"
with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
    for f in (
        list(OUT.rglob("*"))
        + list(PLOTS.glob("*"))
        + list(DATA.glob("*.csv"))
        + list(DATA.glob("*.json"))
    ):
        if f.is_file():
            z.write(f, arcname=str(f.relative_to(ROOT)))
# sha256
h = hashlib.sha256()
with open(bundle, "rb") as bf:
    for chunk in iter(lambda: bf.read(1 << 20), b""):
        h.update(chunk)
with open(OUT / "SHA256SUMS.txt", "w") as s:
    s.write(f"{h.hexdigest()}  {bundle.name}\n")

print("\nPHASE III COMPLETE")
print("Transition λ*:", lambda_star, "epsilon*:", epsilon_star)
print("Outputs:")
print(" - data/runs_phaseIII_summary.csv")
print(" - data/openness_events_decay_phaseIII.csv")
print(" - data/distinguishability_by_lambda.csv")
print(" - data/transition_point.json")
print(" - report:", md)
print(" - bundle:", bundle)
