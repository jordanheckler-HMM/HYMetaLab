# experiments/agent_strategies_runner.py
# Targeted experiments: Do agent strategies help them personally during civilization collapse?
# Outputs: ./discovery_results/agent_strategies/

import json
import math
import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTDIR = os.path.join("discovery_results", "agent_strategies")
os.makedirs(OUTDIR, exist_ok=True)

# -----------------------------
# Aggregate civilization model (compatible with your previous runners):
# - We simulate a large society with sub-groups of agents using different strategies.
# - Global state: tech, cci, inequality, population, coordination (social_weight), goal_diversity
# - A scheduled shock occurs; strategies may activate BEFORE / DURING / AFTER the shock.
# - We track per-strategy "payoffs":
#    - survival_score: proxy for subgroup persistence (resource share * cohesion)
#    - relative_advantage: subgroup resource share vs. its initial share
#    - welfare: combined (survival_score weighted by final tech/cci)
# - Strategies change local & global dynamics following your reports:
#   ALIGN: raises subgroup CCI & local coordination, slightly lowers inequality
#   HOARD: increases subgroup resource share & raises global inequality / lowers coord
#   FRAGMENT: splits into small pocket (reduces exposure to global collapse; helps after shocks)
#   SHARE: reduces inequality globally; slightly hurts short-term subgroup advantage; improves recovery
#   SABOTAGE: increases inequality & lowers coord (bad globally, may benefit small subgroup short-term)
# -----------------------------


@dataclass
class CivParams:
    init_tech: float = 0.9
    init_cci: float = 0.7
    population: int = 300  # complexity band (100, 300, 500)
    goal_diversity: int = 4  # 3-4 ~ optimal
    social_weight: float = 0.6  # 0..1 coordination
    inequality: float = 0.3  # 0.12..0.6 proxy for Gini
    steps: int = 600
    base_growth: float = 0.004
    innovation_rate: float = 0.02


@dataclass
class ShockSpec:
    t: int
    severity: float  # 0..1 (0.5 transition; 0.8 catastrophic)
    kind: str  # "external","internal","combo"


# Strategy names
STRATS = ["ALIGN", "HOARD", "FRAGMENT", "SHARE", "SABOTAGE"]


@dataclass
class StratConfig:
    # Fractions of population using each strategy; sums to <=1.0 (rest = neutral)
    mix: dict[str, float]
    # When strategy activates relative to shock: "BEFORE","DURING","AFTER"
    timing: str


@dataclass
class CivState:
    tech: float
    cci: float
    population: int
    inequality: float
    social_weight: float
    goal_diversity: int
    max_tech_ever: float
    artifacts_score: float
    collapsed: bool
    # resource shares for each strategy subgroup + neutral remainder
    shares: dict[str, float]


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def collapse_risk_prob(
    inequality: float, population: int, social_weight: float, cci: float
) -> float:
    complexity = min(1.0, population / 500.0)
    raw = (inequality * complexity) / max(
        1e-6, (0.25 + 0.75 * social_weight) * (0.3 + 0.7 * cci)
    )
    # bounded logistic
    return clamp(1 / (1 + math.exp(-6 * (raw - 0.5))), 0.0, 1.0)


def apply_shock(state: CivState, shock: ShockSpec):
    sev = shock.severity
    # Damage profiles
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

    # Artifacts accumulate when far below prior peak
    gap = max(0.0, state.max_tech_ever - state.tech)
    state.artifacts_score += gap * (0.5 + 0.5 * sev)

    if sev >= 0.8:
        state.collapsed = True


def step_dynamics(state: CivState, p: CivParams):
    # organic growth
    pop_factor = math.log(max(2, state.population)) / math.log(500)
    tech_growth = p.base_growth * (0.5 + 0.5 * state.cci) * pop_factor
    tech_growth += p.innovation_rate * (0.5 - abs(state.cci - 0.7)) * 0.002
    # inequality drift
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
    # target CCI toward social/inequality/diversity optimum
    diversity_bonus = -abs(state.goal_diversity - 3.5) * 0.02 + 0.03
    target_cci = clamp(
        0.5 * state.social_weight + 0.3 * (1 - state.inequality) + diversity_bonus,
        0.0,
        1.0,
    )
    state.cci = clamp(state.cci + 0.02 * (target_cci - state.cci), 0.0, 1.0)
    state.tech = clamp(state.tech + tech_growth, 0.0, 1.2)
    state.max_tech_ever = max(state.max_tech_ever, state.tech)


def activate_strategies(state: CivState, cfg: StratConfig):
    """
    Apply strategic effects to global state + subgroup resource shares.
    Effects are small per step but cumulative.
    """
    mix = cfg.mix
    # Base multipliers (tuned to be modest but directional)
    align_uplift = 0.006 * mix.get("ALIGN", 0.0)  # raises coordination & CCI
    share_eq = 0.006 * mix.get("SHARE", 0.0)  # lowers inequality slightly
    hoard_ineq = 0.008 * mix.get("HOARD", 0.0)  # raises inequality
    sabotage_hit = 0.008 * mix.get("SABOTAGE", 0.0)  # lowers coord & CCI
    frag_benefit = 0.005 * mix.get(
        "FRAGMENT", 0.0
    )  # reduces exposure to global turbulence

    # Global shifts
    state.social_weight = clamp(
        state.social_weight
        + align_uplift
        - 0.006 * mix.get("HOARD", 0.0)
        - 0.010 * mix.get("SABOTAGE", 0.0),
        0.2,
        0.9,
    )
    state.inequality = clamp(
        state.inequality - share_eq + hoard_ineq + 0.004 * mix.get("SABOTAGE", 0.0),
        0.05,
        0.6,
    )
    state.cci = clamp(
        state.cci + 0.01 * mix.get("ALIGN", 0.0) - 0.008 * mix.get("SABOTAGE", 0.0),
        0.0,
        1.0,
    )

    # Resource share shifts (relative advantage mechanics)
    # Start from current shares
    delta = {
        "ALIGN": +0.004,
        "SHARE": -0.001,  # pays short-term cost
        "HOARD": +0.010,
        "SABOTAGE": +0.006,
        "FRAGMENT": +0.003,  # gradual resilience gain
    }
    # Apply weighted by mix magnitude
    for k, v in delta.items():
        inc = v * mix.get(k, 0.0)
        state.shares[k] = max(0.0, state.shares.get(k, 0.0) + inc)

    # Renormalize shares to cap total <= 1.0 (neutral remainder gets the rest)
    total = sum(state.shares.get(k, 0.0) for k in STRATS)
    if total > 0.95:  # leave at least 5% neutral
        scale = 0.95 / total
        for k in STRATS:
            state.shares[k] *= scale


def simulate_run(seed: int, p: CivParams, shock: ShockSpec, strat: StratConfig):
    random.seed(seed)
    np.random.seed(seed)

    # Initialize neutral remainder
    base_shares = {k: max(0.0, strat.mix.get(k, 0.0)) for k in STRATS}
    total = sum(base_shares.values())
    if total > 0.95:
        scale = 0.95 / total
        for k in base_shares:
            base_shares[k] *= scale

    st = CivState(
        tech=p.init_tech,
        cci=p.init_cci,
        population=p.population,
        inequality=p.inequality,
        social_weight=p.social_weight,
        goal_diversity=p.goal_diversity,
        max_tech_ever=p.init_tech,
        artifacts_score=0.0,
        collapsed=False,
        shares=base_shares,
    )

    history = []
    for t in range(p.steps):
        # Apply strategies BEFORE shock?
        if strat.timing == "BEFORE" and t < shock.t:
            activate_strategies(st, strat)

        # Natural dynamics
        step_dynamics(st, p)

        # Spontaneous internal crisis (stochastic)
        risk = collapse_risk_prob(
            st.inequality, st.population, st.social_weight, st.cci
        )
        if random.random() < 0.0015 + 0.015 * risk:
            apply_shock(
                st,
                ShockSpec(t=t, severity=0.55 + 0.3 * random.random(), kind="internal"),
            )

        # DURING: amplify around shock window
        if strat.timing == "DURING" and (t >= shock.t - 3 and t <= shock.t + 3):
            activate_strategies(st, strat)

        # Scheduled shock
        if t == shock.t:
            apply_shock(st, shock)

        # AFTER: recovery-phase strategy
        if strat.timing == "AFTER" and t > shock.t:
            activate_strategies(st, strat)

        # Post-collapse stall easing
        if st.tech < 0.4 and st.cci < 0.5:
            st.tech = clamp(st.tech + 0.0006 * st.cci)

        history.append(
            {
                "t": t,
                "tech": st.tech,
                "cci": st.cci,
                "inequality": st.inequality,
                "social_weight": st.social_weight,
                "max_tech_ever": st.max_tech_ever,
                "artifacts_score": st.artifacts_score,
                "collapsed_flag": int(st.collapsed),
                **{f"share_{k}": st.shares.get(k, 0.0) for k in STRATS},
            }
        )

    # Compute per-strategy payoffs
    # resource advantage: final share vs initial share
    payoffs = {}
    for k in STRATS:
        init_share = max(1e-6, history[0].get(f"share_{k}", 0.0))
        final_share = history[-1].get(f"share_{k}", 0.0)
        relative_adv = final_share - init_share
        # survival proxy: more resources + higher final cci in subgroup context; fragmentation gains bonus after shock
        survival_score = max(0.0, final_share) * (0.6 + 0.4 * history[-1]["cci"])
        if k == "FRAGMENT":
            # if collapsed at any point, fragmentation gains an extra survival bump (pocket resilience)
            if any(h["collapsed_flag"] for h in history[shock.t :]):
                survival_score *= 1.15
        welfare = survival_score * (0.5 + 0.5 * history[-1]["tech"])
        payoffs[k] = {
            "relative_advantage": relative_adv,
            "survival_score": survival_score,
            "welfare": welfare,
        }

    out = {
        "seed": seed,
        "final": {
            "tech": history[-1]["tech"],
            "cci": history[-1]["cci"],
            "inequality": history[-1]["inequality"],
            "social_weight": history[-1]["social_weight"],
            "regained_prior_peak": float(
                history[-1]["tech"] >= 0.98 * max(h["max_tech_ever"] for h in history)
            ),
            "time_to_80pct_recovery": next(
                (
                    h["t"]
                    for h in history
                    if h["tech"] >= 0.8 * max(hh["max_tech_ever"] for hh in history)
                ),
                None,
            ),
            "artifacts_score": history[-1]["artifacts_score"],
            "collapsed_flag": int(history[-1]["collapsed_flag"]),
        },
        "payoffs": payoffs,
        "history": history,
    }
    return out


def make_sweep():
    seeds = list(range(30))  # 30 per config to keep runtime reasonable
    populations = [100, 300, 500]
    base_params = [
        CivParams(
            init_tech=0.9, init_cci=0.7, inequality=0.28, social_weight=0.65
        ),  # healthy
        CivParams(
            init_tech=0.9, init_cci=0.6, inequality=0.40, social_weight=0.55
        ),  # strained
        CivParams(
            init_tech=0.9, init_cci=0.6, inequality=0.52, social_weight=0.49
        ),  # brittle (Earth-like snapshot)
    ]
    for p in base_params:
        for pop in populations:
            p.population = pop

    shocks = [
        ShockSpec(t=200, severity=0.5, kind="external"),
        ShockSpec(t=200, severity=0.8, kind="combo"),
    ]

    # Strategy mixes (each sums to <=1; remainder is neutral)
    mixes = [
        {"ALIGN": 0.3},  # 30% aligners
        {"HOARD": 0.2},  # 20% hoarders
        {"FRAGMENT": 0.2},  # 20% fragment pockets
        {"SHARE": 0.3},  # 30% redistributors
        {"SABOTAGE": 0.1},  # 10% saboteurs
        {"ALIGN": 0.2, "SHARE": 0.1},  # pro-social blend
        {"HOARD": 0.1, "SABOTAGE": 0.05},  # predatory blend
        {"FRAGMENT": 0.15, "ALIGN": 0.15},  # align + fragment
    ]
    timings = ["BEFORE", "DURING", "AFTER"]

    rows = []
    run_id = 0
    for p in base_params:
        for shock in shocks:
            for mix in mixes:
                for timing in timings:
                    strat = StratConfig(mix=mix, timing=timing)
                    for seed in seeds:
                        out = simulate_run(seed, p, shock, strat)
                        fin = out["final"]

                        # Extract per-strategy payoffs into flat columns
                        payoff_cols = {}
                        for k in STRATS:
                            pk = out["payoffs"][k]
                            payoff_cols[f"{k}_reladv"] = pk["relative_advantage"]
                            payoff_cols[f"{k}_survival"] = pk["survival_score"]
                            payoff_cols[f"{k}_welfare"] = pk["welfare"]

                        rows.append(
                            {
                                "run_id": run_id,
                                "seed": seed,
                                "population": p.population,
                                "init_cci": p.init_cci,
                                "init_ineq": p.inequality,
                                "init_social": p.social_weight,
                                "shock_severity": shock.severity,
                                "shock_kind": shock.kind,
                                "timing": timing,
                                **{f"mix_{k}": mix.get(k, 0.0) for k in STRATS},
                                "final_tech": fin["tech"],
                                "final_cci": fin["cci"],
                                "final_ineq": fin["inequality"],
                                "final_social": fin["social_weight"],
                                "regained_prior_peak": fin["regained_prior_peak"],
                                "time_to_80pct_recovery": fin["time_to_80pct_recovery"],
                                "artifacts_score": fin["artifacts_score"],
                                "collapsed_flag": fin["collapsed_flag"],
                                **payoff_cols,
                            }
                        )
                        run_id += 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "agent_strategies_sweep.csv"), index=False)

    # Summaries
    # 1) Which strategies tend to benefit under each context?
    strategy_cols = [f"{k}_welfare" for k in STRATS]
    by_context = (
        df.groupby(
            ["init_ineq", "init_social", "population", "shock_severity", "timing"]
        )[
            strategy_cols
            + [
                "collapsed_flag",
                "regained_prior_peak",
                "artifacts_score",
                "final_tech",
                "final_cci",
            ]
        ]
        .mean()
        .reset_index()
    )
    by_context.to_csv(os.path.join(OUTDIR, "summary_by_context.csv"), index=False)

    # 2) Mix effectiveness (average welfare per strategy vs. mixes)
    by_mix = (
        df.groupby(
            [
                "mix_ALIGN",
                "mix_HOARD",
                "mix_FRAGMENT",
                "mix_SHARE",
                "mix_SABOTAGE",
                "shock_severity",
                "timing",
            ]
        )[strategy_cols + ["collapsed_flag", "final_tech", "final_cci"]]
        .mean()
        .reset_index()
    )
    by_mix.to_csv(os.path.join(OUTDIR, "summary_by_mix.csv"), index=False)

    # 3) Timing comparison
    by_timing = (
        df.groupby(["timing", "shock_severity"])[
            strategy_cols
            + ["collapsed_flag", "regained_prior_peak", "final_tech", "final_cci"]
        ]
        .mean()
        .reset_index()
    )
    by_timing.to_csv(os.path.join(OUTDIR, "summary_by_timing.csv"), index=False)

    # Plots
    def bar_plot(vals: dict[str, float], title: str, fname: str):
        plt.figure()
        keys = list(vals.keys())
        vals_ = [vals[k] for k in keys]
        plt.bar(keys, vals_)
        plt.title(title)
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, fname), dpi=160)
        plt.close()

    # Global view under catastrophic shocks (severity=0.8), brittle context (Earth-like)
    cat = by_context[
        (by_context["shock_severity"] == 0.8)
        & (by_context["init_ineq"] >= 0.5)
        & (by_context["init_social"] <= 0.5)
    ]
    if not cat.empty:
        mean_welfare = {k.replace("_welfare", ""): cat[k].mean() for k in strategy_cols}
        bar_plot(
            mean_welfare,
            "Mean Strategy Welfare (Catastrophic Shock, Brittle Context)",
            "plot_welfare_catastrophic_brittle.png",
        )

    # Overall mean welfare across all contexts
    overall = {k.replace("_welfare", ""): by_context[k].mean() for k in strategy_cols}
    bar_plot(overall, "Mean Strategy Welfare (Overall)", "plot_welfare_overall.png")

    # Timing effectiveness at severity=0.8
    t80 = by_timing[by_timing["shock_severity"] == 0.8]
    if not t80.empty:
        # Avg of all strategies welfare by timing
        tvals = {}
        for timing in sorted(t80["timing"].unique()):
            sub = t80[t80["timing"] == timing]
            tvals[timing] = sub[strategy_cols].mean(axis=1).mean()
        bar_plot(
            tvals,
            "Average Strategy Welfare by Timing (severity=0.8)",
            "plot_timing_sev0_8.png",
        )

    # Topline JSON
    topline = {
        "n_rows": int(len(df)),
        "overall_strategy_rank_by_welfare": sorted(
            overall.items(), key=lambda x: x[1], reverse=True
        ),
        "brittle_catastrophic_rank_by_welfare": (
            sorted(mean_welfare.items(), key=lambda x: x[1], reverse=True)
            if not cat.empty
            else None
        ),
        "notes": "Welfare ~ survival_score * final_tech. Higher is better for subgroup advantage.",
    }
    with open(os.path.join(OUTDIR, "topline.json"), "w") as f:
        json.dump(topline, f, indent=2)

    # Markdown report
    report = []
    report.append("# Agent Strategies in Collapse — Report\n")
    report.append(
        "**Question:** Can agents benefit personally by using strategies during collapse, and when?\n"
    )
    report.append(
        "**Strategies:** ALIGN (raise CCI), HOARD (resource grab), FRAGMENT (small pockets), SHARE (redistribute), SABOTAGE (degrade coord).\n"
    )
    report.append("**Timing:** BEFORE / DURING / AFTER shock.\n")
    report.append("## Overall Strategy Welfare (higher = better for subgroup)\n")
    report.append("```")
    report.append(pd.DataFrame(overall, index=[0]).to_string(index=False))
    report.append("```")
    if not cat.empty:
        report.append(
            "\n\n## Catastrophic Shock + Brittle Context (Earth-like) — Welfare Ranking\n"
        )
        report.append("```")
        report.append(pd.DataFrame(mean_welfare, index=[0]).to_string(index=False))
        report.append("```")
    report.append(
        "\n\n## Files\n- agent_strategies_sweep.csv\n- summary_by_context.csv\n- summary_by_mix.csv\n- summary_by_timing.csv\n- plot_welfare_overall.png\n- plot_welfare_catastrophic_brittle.png\n- plot_timing_sev0_8.png\n- topline.json\n"
    )
    with open(os.path.join(OUTDIR, "agent_strategies_report.md"), "w") as f:
        f.write("\n".join(report))

    print(f"[OK] Wrote results to: {OUTDIR}")
    return df


if __name__ == "__main__":
    df = make_sweep()
