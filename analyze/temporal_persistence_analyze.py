"""
Analyze script for the temporal_persistence study.

Usage:
  python analyze/temporal_persistence_analyze.py --in <discovery_results/temporal_persistence> --out <outdir>

Produces figures and a short 2-page report with bootstrap CIs.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
import scipy.stats as stats
import seaborn as sns


def fit_exponential(times, y):
    # y = A * exp(-t/tau)
    mask = ~np.isnan(y)
    t = times[mask]
    yy = np.clip(y[mask], 1e-9, 1.0)
    if len(t) < 3:
        return dict(tau=np.nan, R2=np.nan, params=(np.nan, np.nan))
    try:
        logy = np.log(yy)
        A = np.vstack([-t, np.ones_like(t)]).T
        sol, *_ = np.linalg.lstsq(A, logy, rcond=None)
        slope, intercept = sol
        tau = 1.0 / slope if slope != 0 else np.inf
        y_pred = np.exp(intercept + slope * t)
        ss_res = np.sum((yy - y_pred) ** 2)
        ss_tot = np.sum((yy - yy.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return dict(
            tau=float(tau), R2=float(R2), params=(float(intercept), float(slope))
        )
    except Exception:
        return dict(tau=np.nan, R2=np.nan, params=(np.nan, np.nan))


def stretched_exp(t, A, tau, beta):
    return A * np.exp(-((t / tau) ** beta))


def fit_stretched(times, y):
    mask = ~np.isnan(y)
    t = times[mask]
    yy = np.clip(y[mask], 1e-9, 1.0)
    if len(t) < 4:
        return dict(tau=np.nan, beta=np.nan, R2=np.nan, params=None)
    try:
        p0 = [1.0, 10.0, 0.9]
        popt, pcov = opt.curve_fit(stretched_exp, t, yy, p0=p0, maxfev=10000)
        A, tau, beta = popt
        y_pred = stretched_exp(t, *popt)
        ss_res = np.sum((yy - y_pred) ** 2)
        ss_tot = np.sum((yy - yy.mean()) ** 2)
        R2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        return dict(
            tau=float(tau), beta=float(beta), R2=float(R2), params=(A, tau, beta)
        )
    except Exception:
        return dict(tau=np.nan, beta=np.nan, R2=np.nan, params=None)


def bootstrap_ci(data, func, n_boot=800, alpha=0.05, rng=None):
    rng = rng or np.random.RandomState(0)
    stats = []
    n = len(data)
    for i in range(n_boot):
        samp = rng.choice(data, size=n, replace=True)
        stats.append(func(samp))
    lo = np.percentile(stats, 100 * alpha / 2.0)
    hi = np.percentile(stats, 100 * (1 - alpha / 2.0))
    return lo, hi, np.mean(stats)


def quadratic_fit(x, y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 4:
        return None
    coeffs = np.polyfit(x[mask], y[mask], 2)
    # vertex at -b/(2a)
    a, b, c = coeffs
    if a == 0:
        vertex = None
    else:
        vertex = -b / (2 * a)
    return dict(coeffs=coeffs, vertex=vertex)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="indir", required=True)
    parser.add_argument("--out", dest="outdir", required=True)
    parser.add_argument("--bootstrap", dest="bootstrap", type=int, default=800)
    args = parser.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect JSON outputs from runs
    runs = list(indir.rglob("*.json"))
    records = []
    fidelity_curves = []
    for f in runs:
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        # Expect config keys inside j
        cfg = j.get("config", {})
        res = j.get("result", j)
        # support both 'eta' and 'noise_eta'
        eta = float(cfg.get("noise_eta", cfg.get("eta", np.nan)))
        rec = dict(
            path=str(f),
            epsilon=float(cfg.get("epsilon", np.nan)),
            eta=eta,
            rho=float(cfg.get("rho", np.nan)),
            lambda_init=float(cfg.get("lambda_init", np.nan)),
            seed=int(cfg.get("seed", 0)),
            tau_c=float(res.get("tau_c", np.nan)),
            fit_R2=float(res.get("fit_R2", np.nan)),
            t_arrow_var=float(res.get("t_arrow_var", np.nan)),
        )
        records.append(rec)
        # store curves for plotting per-seed at reduced fidelity
        if "fid_curve" in res:
            fidelity_curves.append(dict(config=cfg, fid=np.array(res["fid_curve"])))

    df = pd.DataFrame.from_records(records)
    df.to_csv(outdir / "temporal_persistence_summary.csv", index=False)

    # Regression: tau_c ~ epsilon / eta
    # We'll use epsilon and eta as predictors; create ratio
    df["eps_over_eta"] = df["epsilon"] / (df["eta"] + 1e-9)

    model1 = stats.linregress(df["eps_over_eta"].values, df["tau_c"].values)

    # Regression: t_arrow_var ~ (1 - lambda_est)/epsilon
    # We don't have lambda_est directly at aggregate level; approximate with (1 - lambda_init)
    df["one_minus_lambda_over_eps"] = (1.0 - df["lambda_init"]) / (df["epsilon"] + 1e-9)
    model2 = stats.linregress(
        df["one_minus_lambda_over_eps"].values, df["t_arrow_var"].values
    )

    # Quadratic fit tau_c ~ a*rho^2 + b*rho + c
    q = quadratic_fit(df["rho"].values, df["tau_c"].values)

    # Bootstrap CI for vertex via resampling rows
    rng = np.random.RandomState(0)

    def vertex_of_sample(sample_rows):
        s = df.iloc[sample_rows]
        fit = quadratic_fit(s["rho"].values, s["tau_c"].values)
        return fit["vertex"] if fit and fit["vertex"] is not None else np.nan

    n = len(df)
    verts = []
    for i in range(args.bootstrap):
        rows = rng.randint(0, n, size=n)
        v = vertex_of_sample(rows)
        verts.append(v)
    verts = np.array(verts)
    verts = verts[np.isfinite(verts)]
    v_lo, v_hi = (
        np.percentile(verts, [2.5, 97.5]) if len(verts) > 0 else (np.nan, np.nan)
    )

    # Save intermediate results
    with open(outdir / "temporal_persistence_models.json", "w") as fh:
        json.dump(
            dict(
                model1=dict(
                    slope=model1.slope,
                    intercept=model1.intercept,
                    rvalue=model1.rvalue,
                    pvalue=model1.pvalue,
                ),
                model2=dict(
                    slope=model2.slope,
                    intercept=model2.intercept,
                    rvalue=model2.rvalue,
                    pvalue=model2.pvalue,
                ),
                quadratic=q,
                vertex_ci=(float(v_lo), float(v_hi)),
            ),
            fh,
            indent=2,
        )

    # Figures
    sns.set(style="whitegrid")
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="eps_over_eta", y="tau_c", data=df)
    plt.title("tau_c vs epsilon/eta")
    plt.savefig(outdir / "tau_vs_eps_over_eta.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="one_minus_lambda_over_eps", y="t_arrow_var", data=df)
    plt.title("t_arrow_var vs (1-lambda)/epsilon")
    plt.savefig(outdir / "tarrowvar_vs_one_minus_lambda_over_eps.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="rho", y="tau_c", data=df)
    if q and "coeffs" in q:
        xs = np.linspace(df["rho"].min(), df["rho"].max(), 200)
        a, b, c = q["coeffs"]
        ys = a * xs**2 + b * xs + c
        plt.plot(xs, ys, color="red")
        if q.get("vertex") is not None:
            plt.axvline(
                q["vertex"],
                color="orange",
                linestyle="--",
                label=f'vertex={q["vertex"]:.3f}',
            )
    plt.title("tau_c vs rho (quadratic fit)")
    plt.savefig(outdir / "tau_vs_rho_quadratic.png", dpi=150)
    plt.close()

    # Heatmap tau by epsilon vs eta
    try:
        pivot = df.pivot_table(
            index="eta", columns="epsilon", values="tau_c", aggfunc="mean"
        )
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
        plt.title("tau_c heatmap by epsilon (cols) and eta (rows)")
        plt.savefig(outdir / "heatmap_tau_by_epsilon_eta.png", dpi=150)
        plt.close()
    except Exception:
        pass

    # arrow variance vs (1-lambda)/epsilon
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x="one_minus_lambda_over_eps", y="t_arrow_var", data=df)
    plt.title("t_arrow_var vs (1-lambda)/epsilon")
    plt.savefig(outdir / "arrow_variance_vs_lambda_eps.png", dpi=150)
    plt.close()

    # Save fidelity curves at delta_fidelity sampling
    delta = 10
    try:
        df_curves_dir = outdir / "curves"
        df_curves_dir.mkdir(parents=True, exist_ok=True)
        for item in fidelity_curves:
            cfg = item["config"]
            fid = item["fid"]
            sampled = fid[::delta]
            times = np.arange(len(sampled)) * delta
            plt.figure(figsize=(6, 3))
            plt.plot(times, sampled)
            plt.title(
                f"F(t) seed={cfg.get('seed',0)} eps={cfg.get('epsilon')} eta={cfg.get('noise_eta', cfg.get('eta'))}"
            )
            pngname = df_curves_dir / f"curves_fidelity_seed{cfg.get('seed',0)}.png"
            plt.savefig(pngname, dpi=150)
            plt.close()
    except Exception:
        pass

    # Short 2-page report (plain text + simple markdown)
    report_txt = []
    report_txt.append("Temporal Persistence Study - 2-page report")
    report_txt.append("========================================")
    report_txt.append("\nSummary statistics:")
    report_txt.append(df.describe().to_string())
    report_txt.append("\nModel: tau_c ~ epsilon/eta")
    report_txt.append(
        f"  slope={model1.slope:.5g}, intercept={model1.intercept:.5g}, r={model1.rvalue:.4f}, p={model1.pvalue:.3g}"
    )
    report_txt.append("\nModel: t_arrow_var ~ (1-lambda)/epsilon")
    report_txt.append(
        f"  slope={model2.slope:.5g}, intercept={model2.intercept:.5g}, r={model2.rvalue:.4f}, p={model2.pvalue:.3g}"
    )
    report_txt.append("\nQuadratic fit tau_c ~ a*rho^2 + b*rho + c")
    report_txt.append(
        f'  coeffs={q["coeffs"] if q else None}, vertex={q["vertex"] if q else None}, vertex_CI=({v_lo:.3f},{v_hi:.3f})'
    )

    (outdir / "REPORT.md").write_text("\n\n".join(report_txt))

    print("Analysis complete. Outputs written to", outdir)


if __name__ == "__main__":
    main()
