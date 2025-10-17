# ===========================================================
# phase27b_figure_pack_generator.py
# Purpose: Generate publication-ready figures + summary for the latest VALIDATED dataset
# Output: PNG figures + README.md inside the validated run's /report folder
# ===========================================================
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

# ---------- 0) Locate the most recent VALIDATED entry ----------
ARCHIVE = Path("./project_archive")
summaries = sorted(ARCHIVE.glob("project_summary_*.json"))
if not summaries:
    raise SystemExit("[!] No project_summary_*.json found. Run the organizer first.")

latest = summaries[-1]
data = json.loads(latest.read_text())

validated = [d for d in data if d.get("category") == "validated" and d.get("file")]
if not validated:
    raise SystemExit("[!] No validated datasets in the latest organizer summary.")

# pick the most recent validated file path
v = sorted(validated, key=lambda r: r["file"])[-1]
run_csv_path = Path(v["file"])
run_dir = run_csv_path.parent.parent  # .../data -> run root
report_dir = run_dir / "report"
report_dir.mkdir(parents=True, exist_ok=True)

print(f"[✓] Using validated dataset: {run_csv_path}")

# ---------- 1) Load data ----------
df = pd.read_csv(run_csv_path)

# Harmonize expected columns if needed
col = {c.lower(): c for c in df.columns}


def getcol(name, fallbacks=()):
    cand = [name] + list(fallbacks)
    for c in cand:
        if c in col:
            return col[c]
    return None


c_epsilon = getcol("epsilon")
c_cci = getcol("cci", ("final_cci", "cci_mean"))
c_surv = getcol("survival_rate", ("survival", "avg_survival"))
c_shock = getcol("shock")
c_agents = getcol("agents")
c_domain = getcol("domain")
c_cci_lo = getcol("cci_ci_lo")
c_cci_hi = getcol("cci_ci_hi")
c_surv_lo = getcol("survival_ci_lo")
c_surv_hi = getcol("survival_ci_hi")

req = [c_epsilon, c_cci, c_surv]
if any(x is None for x in req):
    missing = ["epsilon", "CCI", "survival_rate"][
        [c_epsilon, c_cci, c_surv].index(None)
    ]
    raise SystemExit(f"[!] Missing required column: {missing}")

# ---------- 2) Helper: safe group means ----------
group_keys = [k for k in [c_domain, c_agents, c_shock, c_epsilon] if k]
g = df.groupby(group_keys, dropna=False, as_index=False)[[c_cci, c_surv]].mean()


# ---------- 3) Effect sizes (slope vs epsilon) ----------
def slope_ci(x, y, B=1000, seed=0):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.nan)
    b = linregress(x, y).slope
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    boots = []
    for _ in range(B):
        s = rng.choice(idx, size=len(idx), replace=True)
        X = np.vstack([np.ones_like(x[s]), x[s]]).T
        beta = np.linalg.lstsq(X, y[s], rcond=None)[0][1]
        boots.append(beta)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return (float(b), float(lo), float(hi), float(np.mean(boots)))


def effect_table(frame, domain=None):
    rows = []
    if domain and c_domain and domain in frame[c_domain].unique():
        frame = frame[frame[c_domain] == domain]
    for metric, col in [("CCI", c_cci), ("survival", c_surv)]:
        s, lo, hi, bmean = slope_ci(frame[c_epsilon], frame[col])
        rows.append(
            {
                "metric": metric,
                "slope_per_eps": round(s, 4),
                "boot_lo": round(lo, 4),
                "boot_hi": round(hi, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------- 4) Plots ----------
ts = datetime.now().strftime("%Y%m%d_%H%M%S")


def savefig(name):
    p = report_dir / name
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close()
    print(f"[✓] {p}")
    return p


# 4a) CCI vs epsilon (all data)
plt.figure(figsize=(8, 5))
plt.scatter(df[c_epsilon], df[c_cci], s=6)
plt.xlabel("epsilon")
plt.ylabel("CCI")
plt.title("CCI vs epsilon (all samples)")
plt.axhline(0.70, linestyle="--")
cci_scatter = savefig(f"fig_cci_vs_epsilon_{ts}.png")

# 4b) Survival vs epsilon (all data)
plt.figure(figsize=(8, 5))
plt.scatter(df[c_epsilon], df[c_surv], s=6)
plt.xlabel("epsilon")
plt.ylabel("survival_rate")
plt.title("Survival vs epsilon (all samples)")
plt.axhline(0.80, linestyle="--")
surv_scatter = savefig(f"fig_survival_vs_epsilon_{ts}.png")

# 4c) Domain means (if domain exists)
domain_bar = None
if c_domain:
    dmeans = df.groupby(c_domain, as_index=False)[[c_cci, c_surv]].mean()
    plt.figure(figsize=(8, 5))
    x = np.arange(len(dmeans))
    w = 0.35
    plt.bar(x, dmeans[c_cci], width=w, label="CCI")
    plt.bar(x + w, dmeans[c_surv], width=w, label="survival")
    plt.xticks(x + w / 2, dmeans[c_domain])
    plt.axhline(0.70, linestyle="--")
    plt.axhline(0.80, linestyle="--")
    plt.ylabel("mean value")
    plt.title("Domain means (CCI & survival)")
    plt.legend()
    domain_bar = savefig(f"fig_domain_means_{ts}.png")

# 4d) If CI columns exist, plot CCI mean with CI ribbon per epsilon
if c_cci_lo and c_cci_hi:
    gg = df.groupby(c_epsilon, as_index=False)[[c_cci]].mean()
    lo = (
        df.groupby(c_epsilon, as_index=False)[[c_cci_lo]]
        .mean()
        .rename(columns={c_cci_lo: "lo"})
    )
    hi = (
        df.groupby(c_epsilon, as_index=False)[[c_cci_hi]]
        .mean()
        .rename(columns={c_cci_hi: "hi"})
    )
    plt.figure(figsize=(8, 5))
    plt.plot(gg[c_epsilon], gg[c_cci])
    if "lo" in lo and "hi" in hi:
        x = gg[c_epsilon].values
        lo_v = lo["lo"].reindex_like(gg).values
        hi_v = hi["hi"].reindex_like(gg).values
        plt.fill_between(x, lo_v, hi_v, alpha=0.25)
    plt.axhline(0.70, linestyle="--")
    plt.xlabel("epsilon")
    plt.ylabel("CCI (mean, ±CI)")
    plt.title("CCI mean vs epsilon with CI band")
    cci_ci_plot = savefig(f"fig_cci_ci_band_{ts}.png")

# ---------- 5) Effect-size table(s) ----------
overall_eff = effect_table(g)
tables = [overall_eff]
domains_list = list(df[c_domain].unique()) if c_domain else []
for dname in domains_list:
    try:
        tables.append(effect_table(g, domain=dname).assign(domain=dname))
    except Exception:
        pass

eff_csv = report_dir / f"effect_sizes_{ts}.csv"
pd.concat(tables, ignore_index=True).to_csv(eff_csv, index=False)
print(f"[✓] {eff_csv}")

# ---------- 6) Markdown summary ----------
cci_mean = float(pd.to_numeric(df[c_cci], errors="coerce").mean())
surv_mean = float(pd.to_numeric(df[c_surv], errors="coerce").mean())
ci_flag = bool(c_cci_lo and c_cci_hi)

md = f"""# Validated Figure Pack — {ts}

**Dataset:** `{run_csv_path}`  
**Means (all samples):** CCI = {cci_mean:.4f} | Survival = {surv_mean:.4f}  
**CI columns present:** {'yes' if ci_flag else 'no'}  
**Validation thresholds:** CCI ≥ 0.70, Survival ≥ 0.80

## Figures
- CCI vs epsilon: `{cci_scatter.name}`
- Survival vs epsilon: `{surv_scatter.name}`
{"- Domain means: `" + domain_bar.name + "`" if domain_bar else ""}

## Effect sizes (slopes per ε)
Saved to: `{eff_csv.name}`  
Interpretation rule of thumb: slope CIs excluding 0 → robust monotonic effect.

*Auto-generated from the latest VALIDATED dataset.*
"""
(readme_path := report_dir / f"README_figure_pack_{ts}.md").write_text(
    md, encoding="utf-8"
)
print(f"[✓] {readme_path}")

print("\n[Done] Figure pack written to:", report_dir)
