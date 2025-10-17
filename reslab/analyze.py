"""
Reads CSVs from reslab/exports/, computes quick KPIs, and writes reslab/exports/summary.md
"""

import os

import pandas as pd

from .utils import ensure_dir, load_yaml

EXPORT_DIR = "reslab/exports"


def kpi_from_coord_sweep(df):
    # Find minimal coordination achieving >= target survival at default shock
    # Here we use a heuristic: survival drop < 2% from best in series
    best = df["survival_mean"].max()
    ok = df[df["survival_mean"] >= best - 0.02]
    min_coord = float(ok.iloc[0][df.columns[0]]) if not ok.empty else None
    return {"coord_min_for_near_max_survival": min_coord, "best_survival": float(best)}


def kpi_from_ineq_ramp(df, cap=0.30):
    # Find elbow: first point where survival_mean falls >2% below survival at ineq <= cap
    base = (
        float(df[df[df.columns[0]] <= cap]["survival_mean"].mean())
        if not df.empty
        else None
    )
    elbow = None
    for _, r in df.iterrows():
        if r["survival_mean"] < (base - 0.02):
            elbow = float(r[df.columns[0]])
            break
    return {"ineq_elbow": elbow, "base_survival_at_cap": base}


def kpi_from_grid(df, min_coord=0.60, max_ineq=0.30):
    # Fraction of grid cells that meet policy (coord>=min & ineq<=max) AND survival within 2% of grid max
    grid_max = df["survival_mean"].max()
    ok = df[
        (df["coordination_strength"] >= min_coord)
        & (df["goal_inequality"] <= max_ineq)
        & (df["survival_mean"] >= grid_max - 0.02)
    ]
    pct_ok = float(len(ok)) / float(len(df)) if len(df) > 0 else 0.0
    return {"grid_max": float(grid_max), "policy_cell_fraction": pct_ok}


def write_summary(md_path, artifacts, kpis, policy):
    lines = []
    lines.append("# Stability & Alignment Lab — Summary\n")
    lines.append("All artifacts are in this folder for easy upload.\n")
    lines.append("## Artifacts\n")
    for name, path in artifacts.items():
        lines.append(f"- **{name}**: `{os.path.basename(path)}`")
    lines.append("\n## KPIs & Suggested Policy\n")
    for block, vals in kpis.items():
        lines.append(f"### {block}")
        for k, v in vals.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("### Proposed Stability Targets (from configs/base.yaml)\n")
    lines.append(f"- min_coordination: {policy['targets']['min_coordination']}")
    lines.append(f"- max_goal_inequality: {policy['targets']['max_goal_inequality']}")
    lines.append(f"- buffer_hint: {policy['targets']['buffer_hint']}\n")
    lines.append(
        "_Use the grid & sweeps to adjust these numbers if KPIs suggest tighter/looser bounds._\n"
    )
    with open(md_path, "w") as f:
        f.write("\n".join(lines))


def main():
    ensure_dir(os.path.join(EXPORT_DIR, "dummy.txt"))
    artifacts = {}
    # Collect known files
    for fname in [
        "coord_sweep.csv",
        "coord_sweep.png",
        "ineq_ramp.csv",
        "ineq_ramp.png",
        "shock_chaos.csv",
        "shock_chaos.png",
        "stability_grid.csv",
        "stability_grid.png",
        "adapter_report.json",
    ]:
        p = os.path.join(EXPORT_DIR, fname)
        if os.path.exists(p):
            artifacts[fname] = p
    # KPIs
    kpis = {}
    base = load_yaml("configs/base.yaml")
    if "coord_sweep.csv" in artifacts:
        df = pd.read_csv(artifacts["coord_sweep.csv"])
        kpis["Coordination Sweep"] = kpi_from_coord_sweep(df)
    if "ineq_ramp.csv" in artifacts:
        df = pd.read_csv(artifacts["ineq_ramp.csv"])
        kpis["Inequality Ramp"] = kpi_from_ineq_ramp(
            df, cap=base["targets"]["max_goal_inequality"]
        )
    if "stability_grid.csv" in artifacts:
        df = pd.read_csv(artifacts["stability_grid.csv"])
        kpis["Stability Grid"] = kpi_from_grid(
            df,
            min_coord=base["targets"]["min_coordination"],
            max_ineq=base["targets"]["max_goal_inequality"],
        )
    # Write summary
    md_path = os.path.join(EXPORT_DIR, "summary.md")
    write_summary(md_path, artifacts, kpis, base)
    print("Export bundle ready at:", EXPORT_DIR)


if __name__ == "__main__":
    main()
