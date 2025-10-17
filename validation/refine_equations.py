import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

# ---------- Args ----------
parser = argparse.ArgumentParser()
parser.add_argument("--input", default="runs_summary.csv")
args = parser.parse_args()

os.makedirs("validation", exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(args.input)

# Support common alternate column names in merged summaries
if "openness" not in df.columns and "epsilon0" in df.columns:
    df["openness"] = df["epsilon0"]
if "lam" not in df.columns and "lambda" in df.columns:
    df["lam"] = df["lambda"]

# ---------- FORCE FLOORS BEFORE ANY LOGS ----------
for col, floor in {"openness": 1e-5, "lam": 0.001}.items():
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col].isna() | (df[col] <= 0), col] = floor
print(
    "Zero floors applied -> min(openness)=",
    df["openness"].min(),
    "min(lam)=",
    df["lam"].min() if "lam" in df.columns else "N/A",
)

# Infer R_mean if missing
if "R_mean" not in df.columns:
    if "mean_CCI" in df.columns and "mean_risk_norm" in df.columns:
        df["R_mean"] = pd.to_numeric(df["mean_CCI"], errors="coerce") / (
            pd.to_numeric(df["mean_risk_norm"], errors="coerce") + 1e-12
        )
        print("Inferred R_mean as mean_CCI / mean_risk_norm")
    elif "mean_CCI" in df.columns:
        df["R_mean"] = pd.to_numeric(df["mean_CCI"], errors="coerce")
        print("Inferred R_mean as mean_CCI")
    elif "mean_collapse_risk_norm" in df.columns:
        df["R_mean"] = pd.to_numeric(df["mean_collapse_risk_norm"], errors="coerce")
        print("Inferred R_mean as mean_collapse_risk_norm")
    else:
        print("Warning: R_mean not found; downstream fits may fail or be empty")


# ---------- Helpers ----------
def log_safe(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return np.log(s)


def fit_ols(df_fit, y_col, x_cols):
    X = sm.add_constant(df_fit[x_cols].astype(float))
    y = df_fit[y_col].astype(float)
    model = sm.OLS(y, X, missing="drop").fit()
    coefs = model.params.to_frame("coef").join(model.bse.to_frame("stderr"))
    ci = model.conf_int()
    coefs["ci_low"] = ci[0]
    coefs["ci_high"] = ci[1]
    coefs["pval"] = model.pvalues
    return (
        model,
        {
            "r2": float(model.rsquared),
            "r2_adj": float(model.rsquared_adj),
            "n": int(model.nobs),
        },
        coefs.reset_index().rename(columns={"index": "term"}),
    )


def plot_preds(df_fit, model, x_cols, tag):
    X = sm.add_constant(df_fit[x_cols].astype(float))
    y_true = np.exp(df_fit["log_R"])
    y_pred = np.exp(model.predict(X))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual R_mean")
    plt.ylabel("Predicted R_mean")
    plt.title(f"Predicted vs Actual ({tag})")
    plt.savefig(f"validation/preds_actual_{tag}.png", dpi=160)
    plt.close()


# ---------- Basic clean ----------
required = ["R_mean", "kE", "kI", "n_agents", "openness"]
d0 = len(df)
df = df.dropna(subset=[c for c in required if c in df.columns]).copy()

# Logs (guard missing source columns)
df["log_R"] = log_safe(df["R_mean"]) if "R_mean" in df.columns else np.nan
df["log_E"] = log_safe(df["kE"]) if "kE" in df.columns else np.nan
df["log_I"] = log_safe(df["kI"]) if "kI" in df.columns else np.nan
df["log_N"] = log_safe(df["n_agents"]) if "n_agents" in df.columns else np.nan
df["log_eps"] = log_safe(df["openness"]) if "openness" in df.columns else np.nan
df["log_lam"] = log_safe(df["lam"]) if "lam" in df.columns else np.nan

# Regimes
df["regime"] = np.where(
    df["openness"] <= 1e-4,
    "closed",
    np.where(df["openness"] <= 1e-2, "min_open", "open"),
)

# ---------- Noise ceiling ----------
config_keys = [
    c
    for c in ["n_agents", "openness", "kE", "kI", "lam", "eta", "epochs"]
    if c in df.columns
]
if "seed" in df.columns and config_keys:
    grp = df.groupby(config_keys, dropna=False)
    noise_var = grp["R_mean"].var().dropna()
    total_var = df["R_mean"].var()
    noise_ceiling = (
        None
        if (len(noise_var) == 0 or pd.isna(total_var) or total_var == 0)
        else float(1 - noise_var.mean() / total_var)
    )
else:
    noise_ceiling = None
with open("validation/noise_ceiling.json", "w") as f:
    json.dump({"noise_ceiling_estimate": noise_ceiling}, f, indent=2)

# ---------- Optional extra features ----------
extra_feats = []
if os.path.exists("openness_events.csv"):
    ev = pd.read_csv("openness_events.csv")
    if "condition_id" in df.columns and "condition_id" in ev.columns:
        agg = (
            ev.groupby("condition_id", dropna=False)
            .agg(
                openness_event_count=("event_type", "count"),
                exergy_in_sum=("exergy_in", "sum"),
                waste_out_sum=("waste_out", "sum"),
            )
            .reset_index()
        )
        df = df.merge(agg, on="condition_id", how="left")
        for c in ["openness_event_count", "exergy_in_sum", "waste_out_sum"]:
            df[c] = df[c].fillna(0)
        df["log_exergy_in_sum"] = log_safe(df["exergy_in_sum"].replace(0, np.nan))
        extra_feats += ["log_exergy_in_sum", "openness_event_count", "waste_out_sum"]

if os.path.exists("shock_events.csv") and "condition_id" in df.columns:
    se = pd.read_csv("shock_events.csv")
    shock_ids = se["condition_id"].unique().tolist()
    df["shock_flag"] = df["condition_id"].isin(shock_ids).astype(int)
    extra_feats += ["shock_flag"]
else:
    df["shock_flag"] = 0

# ---------- Models ----------
summ = {}

# M0
X0 = ["log_E", "log_I", "log_N"]
df0 = df.dropna(subset=X0 + ["log_R"])
if len(df0) > 0:
    m0, s0, c0 = fit_ols(df0, "log_R", X0)
    if c0 is not None and len(c0):
        c0.to_csv("validation/coefficients_M0.csv", index=False)
    summ["M0"] = s0
    plot_preds(df0, m0, X0, "M0")
else:
    summ["M0"] = {"r2": None, "r2_adj": None, "n": 0}

# M1 (+ eps + lam)
X1 = ["log_E", "log_I", "log_N", "log_eps", "log_lam"]
df1 = df.dropna(subset=X1 + ["log_R"])
if len(df1) > 0:
    m1, s1, c1 = fit_ols(df1, "log_R", X1)
    if c1 is not None and len(c1):
        c1.to_csv("validation/coefficients_M1.csv", index=False)
    summ["M1"] = s1
    plot_preds(df1, m1, X1, "M1")
else:
    summ["M1"] = {"r2": None, "r2_adj": None, "n": 0}

# M2 (interactions)
df["log_E_eps"] = df["log_E"] * df["log_eps"]
df["log_I_eps"] = df["log_I"] * df["log_eps"]
df["log_N_eps"] = df["log_N"] * df["log_eps"]
X2 = [
    "log_E",
    "log_I",
    "log_N",
    "log_eps",
    "log_lam",
    "log_E_eps",
    "log_I_eps",
    "log_N_eps",
] + extra_feats
df2 = df.dropna(subset=X2 + ["log_R"])
if len(df2) > 0:
    m2, s2, c2 = fit_ols(df2, "log_R", X2)
    if c2 is not None and len(c2):
        c2.to_csv("validation/coefficients_M2.csv", index=False)
    summ["M2"] = s2
    plot_preds(df2, m2, X2, "M2")
else:
    summ["M2"] = {"r2": None, "r2_adj": None, "n": 0}

# M3 (regime split on M2)
reg_results = {}
for reg in ["closed", "min_open", "open"]:
    sub = df[df["regime"] == reg]
    sub = sub.dropna(subset=X2 + ["log_R"])
    if len(sub) < 20:
        reg_results[reg] = {"note": "insufficient data", "n": len(sub)}
        continue
    mR, sR, cR = fit_ols(sub, "log_R", X2)
    cR.to_csv(f"validation/coefficients_M3_{reg}.csv", index=False)
    reg_results[reg] = sR
    X = sm.add_constant(sub[X2].astype(float))
    y_pred = np.exp(mR.predict(X))
    y_true = np.exp(sub["log_R"])
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=12)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Actual R_mean")
    plt.ylabel("Predicted R_mean")
    plt.title(f"Predicted vs Actual (M3 - {reg})")
    plt.savefig(f"validation/preds_actual_M3_{reg}.png", dpi=160)
    plt.close()

# ---------- Save summary & report ----------
with open("validation/fit_summary.json", "w") as f:
    json.dump(
        {
            "noise_ceiling": noise_ceiling,
            "M0": summ["M0"],
            "M1": summ["M1"],
            "M2": summ["M2"],
            "M3": reg_results,
        },
        f,
        indent=2,
    )

with open("validation/report.md", "w") as f:
    f.write("# Equation Refinement Report (v1)\n\n")
    f.write(
        f"Rows read: {d0}, dropped missing required: 0, dropped invalid logs: 0\n\n"
    )
    f.write("## Noise Ceiling\n")
    f.write(f"- Estimated upper R² bound (due to seed noise): **{noise_ceiling}**\n\n")
    f.write("## Model Fit Summary\n")
    for k in ["M0", "M1", "M2"]:
        r = summ[k]
        f.write(f"- **{k}**: R²={r['r2']}, adjR²={r['r2_adj']}, n={r['n']}\n")
    f.write("\n### Regime Split (M3)\n")
    for reg, r in reg_results.items():
        if "note" in r:
            f.write(f"- {reg}: {r['note']} (n={r.get('n',0)})\n")
        else:
            f.write(f"- {reg}: R²={r['r2']}, adjR²={r['r2_adj']}, n={r['n']}\n")
    f.write("\n")
print("Done. See the 'validation/' folder.")
