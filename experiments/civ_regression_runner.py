# experiments/civ_regression_runner.py
# Civilization Regression Plausibility Experiments
# Purpose: Test whether an advanced civilization can catastrophically regress (and leave "pyramid-like" artifacts)
# Output folder: ./discovery_results/civ_regression/

import json
import math
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = os.path.join("discovery_results", "civ_regression")
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Model Assumptions (grounded in your reports):
# - Collapse risk ~ (Inequality × Complexity) / Coordination   [Theme 9]
# - Constructive vs Destructive shocks threshold near 50%       [Theme 2]
# - High CCI increases survival thresholds & resilience         [Theme 3, 9]
# - Recovery can stall → long plateau below prior tech          [Theme 1, 2]
# We implement a simple, transparent simulator so the pipeline runs even
# if external sim modules are unavailable. If your internal modules exist,
# you can swap the engine with them later.
# -----------------------------


@dataclass
class CivParams:
    init_tech: float = 0.9  # 0-1
    init_cci: float = 0.9  # 0-1 (collective consciousness / coordination quality)
    population: int = 300  # 100, 300, 500
    goal_diversity: int = 4  # 1-5 (optimal ~3-4 per Theme 9)
    social_weight: float = 0.7  # coordination strength 0-1 (optimal 0.6-0.8)
    base_growth: float = 0.004  # baseline tech growth per step
    innovation_rate: float = 0.02  # prevents stagnation; too high -> noise
    inequality: float = 0.24  # proxy for Gini; > 0.30 becomes fragile
    steps: int = 1000  # long horizon for "dark age" plateaus


@dataclass
class ShockSpec:
    t: int
    severity: float  # 0-1 (0.5 ~ transition, 0.8-0.9 catastrophic)
    kind: str  # "external", "internal", "combo"


@dataclass
class CivState:
    tech: float
    cci: float
    population: int
    inequality: float
    social_weight: float
    goal_diversity: int
    max_tech_ever: float
    artifacts_score: float  # accumulates when tech falls far below past peak
    collapsed: bool  # flag if major collapse occurred


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def collapse_risk_prob(
    inequality: float, population: int, social_weight: float, cci: float
) -> float:
    # Complexity term: larger pops more complex; normalize ~ [0,1] around 500
    complexity = min(1.0, population / 500.0)
    # Baseline risk per Theme 9 law, reduced by coordination & CCI
    raw = (inequality * complexity) / max(
        1e-6, (0.25 + 0.75 * social_weight) * (0.3 + 0.7 * cci)
    )
    # Smooth to probability via bounded logistic
    return clamp(1 / (1 + math.exp(-6 * (raw - 0.5))), 0.0, 1.0)


def step_dynamics(state: CivState, p: CivParams):
    # Tech growth driven by CCI, scale with info processing ~ log(pop)
    pop_factor = math.log(max(2, state.population)) / math.log(500)  # ~0..1+
    tech_growth = p.base_growth * (0.5 + 0.5 * state.cci) * pop_factor
    tech_growth += (
        p.innovation_rate * (0.5 - abs(state.cci - 0.7)) * 0.002
    )  # innovation sweet spot near ~0.7

    # Inequality drift: increases with tech & size; decreases with social_weight/CCI
    ineq_drift = (
        0.0005
        * (0.2 + 0.8 * state.tech)
        * (0.3 + 0.7 * min(1.0, state.population / 300))
    )
    ineq_drift *= (1.0 - 0.6 * state.social_weight) * (1.0 - 0.5 * state.cci)
    state.inequality = clamp(
        state.inequality + ineq_drift - 0.00025 * state.social_weight * state.cci,
        0.05,
        0.6,
    )

    # Passive re-calibration: CCI gravitates toward coordination quality & diversity optimum (3-4 goals best)
    diversity_bonus = -abs(state.goal_diversity - 3.5) * 0.02 + 0.03  # peak near 3-4
    target_cci = clamp(
        0.5 * state.social_weight + 0.3 * (1 - state.inequality) + diversity_bonus,
        0.0,
        1.0,
    )
    state.cci = clamp(state.cci + 0.02 * (target_cci - state.cci), 0.0, 1.0)

    # Tech applies growth unless in a post-collapse stall
    state.tech = clamp(
        state.tech + tech_growth, 0.0, 1.2
    )  # allow slight >1 for "golden age" peaks
    state.max_tech_ever = max(state.max_tech_ever, state.tech)


def apply_shock(state: CivState, shock: ShockSpec):
    # Map severity to immediate damage profiles
    sev = shock.severity
    if shock.kind == "external":
        tech_drop = 0.25 * sev + 0.05
        cci_drop = 0.20 * sev + 0.05
        ineq_up = 0.05 * sev + 0.01
    elif shock.kind == "internal":
        tech_drop = 0.18 * sev + 0.03
        cci_drop = 0.26 * sev + 0.06
        ineq_up = 0.08 * sev + 0.02
    else:  # combo
        tech_drop = 0.33 * sev + 0.07
        cci_drop = 0.33 * sev + 0.07
        ineq_up = 0.10 * sev + 0.03

    state.tech = clamp(state.tech * (1 - tech_drop))
    state.cci = clamp(state.cci * (1 - cci_drop))
    state.inequality = clamp(state.inequality + ineq_up, 0.05, 0.6)

    # Residual artifacts: if tech falls far below previous peak, we record a lasting artifact score
    gap = max(0.0, state.max_tech_ever - state.tech)
    state.artifacts_score += gap * (0.5 + 0.5 * sev)

    # Collapse flag for catastrophic events (severity ≳ 0.8 aligns with Theme 2 destructive zone)
    if sev >= 0.8:
        state.collapsed = True


def simulate_run(seed: int, params: CivParams, shocks: list[ShockSpec]) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    st = CivState(
        tech=params.init_tech,
        cci=params.init_cci,
        population=params.population,
        inequality=params.inequality,
        social_weight=params.social_weight,
        goal_diversity=params.goal_diversity,
        max_tech_ever=params.init_tech,
        artifacts_score=0.0,
        collapsed=False,
    )

    history = []
    shock_map = {s.t: s for s in shocks}
    for t in range(params.steps):
        # Natural dynamics
        step_dynamics(st, params)

        # Stochastic collapse kicker per Theme 9 (depends on inequality/coordination/cci)
        risk = collapse_risk_prob(
            st.inequality, st.population, st.social_weight, st.cci
        )
        if random.random() < 0.002 + 0.02 * risk:
            # spontaneous internal crisis
            apply_shock(
                st,
                ShockSpec(t=t, severity=0.65 + 0.2 * random.random(), kind="internal"),
            )

        # Scheduled shock
        if t in shock_map:
            apply_shock(st, shock_map[t])

        # Post-collapse stall: if tech < 0.4 and CCI < 0.5, recovery is slow (dark age plateau)
        if st.tech < 0.4 and st.cci < 0.5:
            st.tech = clamp(st.tech + 0.0005 * st.cci)

        history.append(
            {
                "t": t,
                "tech": st.tech,
                "cci": st.cci,
                "inequality": st.inequality,
                "max_tech_ever": st.max_tech_ever,
                "artifacts_score": st.artifacts_score,
                "collapsed_flag": int(st.collapsed),
            }
        )

    out = {
        "seed": seed,
        "final": {
            "tech": st.tech,
            "cci": st.cci,
            "inequality": st.inequality,
            "regained_prior_peak": float(st.tech >= 0.98 * st.max_tech_ever),
            "time_to_80pct_recovery": next(
                (
                    h["t"]
                    for h in history
                    if h["tech"] >= 0.8 * max(hh["max_tech_ever"] for hh in history)
                ),
                None,
            ),
            "artifacts_score": st.artifacts_score,
            "collapsed_flag": int(st.collapsed),
        },
        "history": history,
    }
    return out


def make_sweep():
    seeds = list(range(50))
    populations = [100, 300, 500]
    goal_diversities = [2, 3, 4]  # test brittle vs optimal vs fragmented edge
    social_weights = [0.5, 0.7, 0.8]  # coordination strength
    shock_severities = [0.5, 0.8, 0.9]  # transition vs destructive/catastrophic
    shock_kinds = ["external", "internal", "combo"]
    shock_times = [100, 200]  # time-of-catastrophe sensitivity

    rows = []
    run_ix = 0
    for pop in populations:
        for gd in goal_diversities:
            for sw in social_weights:
                for sev in shock_severities:
                    for kind in shock_kinds:
                        for stime in shock_times:
                            for seed in seeds:
                                p = CivParams(
                                    population=pop, goal_diversity=gd, social_weight=sw
                                )
                                shocks = [ShockSpec(t=stime, severity=sev, kind=kind)]
                                out = simulate_run(seed, p, shocks)
                                fin = out["final"]
                                rows.append(
                                    {
                                        "run_id": run_ix,
                                        "seed": seed,
                                        "population": pop,
                                        "goal_diversity": gd,
                                        "social_weight": sw,
                                        "shock_severity": sev,
                                        "shock_kind": kind,
                                        "shock_time": stime,
                                        "final_tech": fin["tech"],
                                        "final_cci": fin["cci"],
                                        "final_inequality": fin["inequality"],
                                        "regained_prior_peak": fin[
                                            "regained_prior_peak"
                                        ],
                                        "time_to_80pct_recovery": fin[
                                            "time_to_80pct_recovery"
                                        ],
                                        "artifacts_score": fin["artifacts_score"],
                                        "collapsed_flag": fin["collapsed_flag"],
                                    }
                                )
                                run_ix += 1

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "civ_regression_sweep.csv")
    df.to_csv(csv_path, index=False)

    # Aggregate answers to THE question
    # 1) Regression Plausibility: % runs that end with final_tech << past peak (never recovered)
    df["never_recovered"] = ~(df["regained_prior_peak"].astype(bool))
    df["catastrophic_zone"] = df["shock_severity"] >= 0.8
    df["plausible_pyramids"] = (df["never_recovered"]) & (
        df["artifacts_score"] > df["artifacts_score"].quantile(0.66)
    )

    # Grouped summaries
    by_sev_kind = (
        df.groupby(["shock_severity", "shock_kind"])
        .agg(
            runs=("run_id", "count"),
            pct_never_recovered=("never_recovered", "mean"),
            pct_plausible_pyramids=("plausible_pyramids", "mean"),
            mean_artifact_score=("artifacts_score", "mean"),
            mean_time_to_80pct=("time_to_80pct_recovery", "mean"),
            mean_final_tech=("final_tech", "mean"),
            mean_final_cci=("final_cci", "mean"),
        )
        .reset_index()
    )

    by_context = (
        df.groupby(["population", "goal_diversity", "social_weight", "shock_severity"])
        .agg(
            runs=("run_id", "count"),
            pct_never_recovered=("never_recovered", "mean"),
            pct_plausible_pyramids=("plausible_pyramids", "mean"),
            mean_final_tech=("final_tech", "mean"),
            mean_final_cci=("final_cci", "mean"),
        )
        .reset_index()
    )

    by_sev_kind.to_csv(
        os.path.join(OUTDIR, "summary_by_severity_kind.csv"), index=False
    )
    by_context.to_csv(os.path.join(OUTDIR, "summary_by_context.csv"), index=False)

    # Plots
    plt.figure()
    for sev in sorted(df["shock_severity"].unique()):
        subset = by_sev_kind[by_sev_kind["shock_severity"] == sev]
        x = np.arange(len(subset["shock_kind"]))
        y = subset["pct_plausible_pyramids"].values
        plt.plot(x, y, marker="o", label=f"severity={sev}")
    plt.xticks([0, 1, 2], ["external", "internal", "combo"])
    plt.ylabel("P(plausible pyramids)")
    plt.title("Plausible-Artifact Outcomes by Shock Type/Severity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_plausible_pyramids.png"), dpi=160)
    plt.close()

    plt.figure()
    for sev in sorted(df["shock_severity"].unique()):
        subset = by_sev_kind[by_sev_kind["shock_severity"] == sev]
        x = np.arange(len(subset["shock_kind"]))
        y = subset["pct_never_recovered"].values
        plt.plot(x, y, marker="o", label=f"severity={sev}")
    plt.xticks([0, 1, 2], ["external", "internal", "combo"])
    plt.ylabel("P(never recovered)")
    plt.title("Non-Recovery Probability by Shock Type/Severity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "plot_never_recovered.png"), dpi=160)
    plt.close()

    # Heatmap-like pivot for context factors
    piv = by_context.pivot_table(
        index=["population", "goal_diversity"],
        columns="social_weight",
        values="pct_plausible_pyramids",
        aggfunc="mean",
    )
    piv.to_csv(os.path.join(OUTDIR, "heatmap_data_context_pyramids.csv"))

    # Markdown report
    report = []
    report.append("# Civilization Regression Plausibility Report\n")
    report.append(
        "**Question:** Could an advanced civilization catastrophically regress and later societies inherit large artifacts (e.g., 'pyramids') they can't replicate?\n"
    )
    report.append(
        "**Method:** Multi-factor sweep over population, goal diversity, coordination (social_weight), shock type, severity, and timing with 50 seeds each. Metrics: recovery, final tech/CCI, artifacts score.\n"
    )
    report.append("## Key Findings (Aggregates)\n")
    report.append("```")
    report.append(by_sev_kind.to_string(index=False))
    report.append("```")
    report.append("\n\n### Context Sensitivity\n")
    report.append("```")
    report.append(by_context.head(30).to_string(index=False))
    report.append("```")
    report.append("\n\n## Figures\n")
    report.append(
        "- `plot_plausible_pyramids.png` — Probability of leaving high-contrast artifacts vs. shock type/severity\n"
    )
    report.append(
        "- `plot_never_recovered.png` — Probability of never regaining prior peak\n"
    )
    report_path = os.path.join(OUTDIR, "civ_regression_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    # Save a compact JSON with topline stats for quick preview
    topline = {
        "total_runs": int(len(df)),
        "plausibility_overall": float(df["plausible_pyramids"].mean()),
        "never_recovered_overall": float(df["never_recovered"].mean()),
        "catastrophic_zone_plausibility": float(
            df[df["catastrophic_zone"]]["plausible_pyramids"].mean()
        ),
        "best_resilience_context_hint": by_context.sort_values("pct_never_recovered")
        .head(1)
        .to_dict(orient="records"),
        "worst_context_hint": by_context.sort_values(
            "pct_never_recovered", ascending=False
        )
        .head(1)
        .to_dict(orient="records"),
    }
    with open(os.path.join(OUTDIR, "topline.json"), "w") as f:
        json.dump(topline, f, indent=2)

    print(f"[OK] Wrote results to: {OUTDIR}")
    return df


if __name__ == "__main__":
    df = make_sweep()
