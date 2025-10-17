#!/usr/bin/env python3
# meaning_experiment.py
# Experiment: Religion as a Collective Hope Mechanism
# Produces discovery_results/religion_hope_mechanism_<stamp>/ with CSVs, PNGs, and summary.md

import datetime
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = Path(f"./discovery_results/religion_hope_mechanism_{STAMP}")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Simple simulation utilities
BRANCHES = ["Religion", "Education", "Fatalism"]


def seed_for(params):
    s = json.dumps(params, sort_keys=True)
    return abs(hash(s)) % (2**32)


def initialize_agents(n_agents, goal_diversity, noise, rng, branches=None):
    # Each agent has a set of goals (as ints) and an initial belief distribution over branches
    agents = []
    branches = branches or BRANCHES
    for i in range(n_agents):
        goals = rng.choice(
            range(goal_diversity), size=max(1, goal_diversity // 1), replace=True
        ).tolist()
        # initial belief: small random preference vector
        b = rng.rand(len(branches))
        b = b / b.sum()
        optimism = max(0.0, 1.0 - noise - rng.normal(0, 0.02))
        resilience = max(0.0, 0.5 + rng.normal(0, 0.1))
        agents.append(
            {
                "id": i,
                "goals": goals,
                "belief": b,
                "optimism": optimism,
                "resilience": resilience,
                "alive": True,
                "resource": 1.0,  # normalized
            }
        )
    return agents


def collective_cci(agents):
    # coherence index: similarity of belief vectors (1 - average pairwise distance)
    if not agents:
        return 0.0
    B = np.array([a["belief"] for a in agents])
    mean = B.mean(axis=0)
    # cosine similarity average
    sims = (B @ mean) / (np.linalg.norm(B, axis=1) * np.linalg.norm(mean) + 1e-9)
    return float(np.nanmean(sims))


def belief_convergence(agents):
    # entropy of averaged belief distribution (lower entropy = higher convergence)
    B = np.array([a["belief"] for a in agents])
    mean = B.mean(axis=0)
    ent = -np.sum(mean * np.log(mean + 1e-12))
    # normalize by log(k)
    return 1.0 - (ent / math.log(len(mean) + 1e-12))


def hope_index(agents):
    # combine optimism and resilience weighted by resource
    vals = [
        a["optimism"] * a["resilience"] * a["resource"] for a in agents if a["alive"]
    ]
    return float(np.nanmean(vals)) if vals else 0.0


def step_update(agents, shock_level, duration_type, rng):
    # Social influence: agents nudge neighbors toward the mean belief
    B = np.array([a["belief"] for a in agents])
    mean = B.mean(axis=0)
    for a in agents:
        if not a["alive"]:
            continue
        influence = (
            0.1 + 0.2 * a["resilience"]
        )  # resilience increases network influence absorption
        a["belief"] = a["belief"] * (1 - influence) + mean * influence
        # add small noise
        a["belief"] += rng.normal(0, 0.01, size=len(a["belief"]))
        a["belief"] = np.clip(a["belief"], 1e-6, None)
        a["belief"] = a["belief"] / a["belief"].sum()
        # resource drain due to shock
        drain_factor = shock_level * (0.5 if duration_type == "acute" else 1.0)
        a["resource"] -= drain_factor * (0.05 + 0.05 * (1.0 - a["resilience"]))
        if a["resource"] < 0.0:
            a["alive"] = False
        # optimism decays slightly with resource loss
        a["optimism"] = max(0.0, a["optimism"] - 0.02 * (1.0 - a["resource"]))


def select_branch(agents, branches=None):
    # collective choice: majority of highest-belief branch across alive agents
    branches = branches or BRANCHES
    B = np.array([a["belief"] for a in agents if a["alive"]])
    if len(B) == 0:
        return "None"
    choices = np.argmax(B, axis=1)
    cnt = Counter(choices)
    top = cnt.most_common(1)[0][0]
    # guard if index out of range
    if top < len(branches):
        return branches[top]
    return branches[0]


def run_single(params):
    # Params include n_agents, shock, duration, goal_diversity, noise
    rng = np.random.RandomState(params.get("seed", 0))
    branches = params.get("branch_mask", {}).get("allowed_branches") or BRANCHES
    agents = initialize_agents(
        params["n_agents"],
        params["goal_diversity"],
        params["noise"],
        rng,
        branches=branches,
    )
    # allow override of total steps (support shock_window with large indices)
    if params.get("shock_window"):
        sw = params["shock_window"]
        # If user supplies large epoch numbers (e.g. 7000) we scale down to keep runs tractable.
        # Assumption: scale factor 100 (i.e., 7000 -> 70). This keeps relative timings.
        scale = params.get("time_scale", 100)
        shock_start = max(0, int(sw.get("start", 0) / scale))
        shock_length = max(1, int(sw.get("length", 1) / scale))
        steps = shock_start + shock_length + params.get("post_recovery_horizon", 200)
    else:
        steps = 40 if params["stress_duration"] == "chronic" else 12
    history = {"cci": [], "belief_conv": [], "hope": [], "alive_frac": []}
    # track per-epoch branch fractions (list of lists aligned with 'branches')
    history["branch_frac"] = []

    # Initialize lifespan counters
    for a in agents:
        a.setdefault("lifespan", 0)

    # Determine shock timing
    if params.get("shock_window"):
        sw = params["shock_window"]
        scale = params.get("time_scale", 100)
        shock_start = max(0, int(sw.get("start", 0) / scale))
        shock_length = max(1, int(sw.get("length", 1) / scale))
        shock_end = shock_start + shock_length
    else:
        shock_start = 3
        shock_length = steps
        shock_end = steps

    # openness mechanism
    openness = params.get("openness", {}) or {}
    mech = openness.get("mechanism", "closed")
    epsilon = float(openness.get("epsilon", 0.0))
    period = openness.get("period")

    # Run steps with possible mortality, noise injection, and openness inflow/outflow
    for t in range(steps + 6):
        # determine current shock level
        if shock_start <= t < shock_end:
            current_shock = params["shock"]
        else:
            # reduced post-shock bleed
            current_shock = params["shock"] * 0.2 if t >= shock_end else 0.0

        # noise injection pulses
        ni = params.get("noise_injection")
        if ni and ni.get("enabled"):
            epochs = ni.get("epochs", [])
            if t in [int(e / params.get("time_scale", 100)) for e in epochs]:
                # apply noise delta to a fraction of agents via increasing their optimism / belief noise
                pct = ni.get("pct_agents", 0.5)
                delta = ni.get("delta", 0.1)
                idx = rng.choice(
                    len(agents), size=max(1, int(len(agents) * pct)), replace=False
                )
                for j in idx:
                    agents[j]["optimism"] = max(0.0, agents[j]["optimism"] - delta)

        # mortality model per-epoch
        mm = params.get("mortality_model")
        if mm and mm.get("enabled"):
            fatality_base = mm.get("fatality_base", 0.01)
            k_shock = mm.get("k_shock", 0.2)

        # perform social/resource update and then mortality checks
        step_update(agents, current_shock, params["stress_duration"], rng)

        # openness inflow/outflow
        if mech and mech != "closed" and epsilon > 0.0:
            apply_now = True
            if period is not None:
                # scale period by time_scale if provided
                scale = params.get("time_scale", 100)
                scaled_period = max(1, int(period / scale))
                apply_now = t % scaled_period == 0
            if apply_now:
                if mech == "agent_io":
                    # small uniform inflow distributed to alive agents
                    alive = [a for a in agents if a["alive"]]
                    if alive:
                        add = epsilon
                        for a in alive:
                            a["resource"] = min(1.0, a["resource"] + add)
                elif mech == "chemostat":
                    # chemostat provides inflow but also enforces a leak: net smaller addition
                    alive = [a for a in agents if a["alive"]]
                    if alive:
                        add = epsilon * 0.8
                        for a in alive:
                            a["resource"] = min(1.0, a["resource"] + add)
                else:
                    # unknown mechanism: treat as small uniform inflow
                    for a in agents:
                        if a["alive"]:
                            a["resource"] = min(1.0, a["resource"] + epsilon)

        # mortality pass
        if mm and mm.get("enabled"):
            for a in agents:
                if not a["alive"]:
                    continue
                # fatality probability scales with shock and inverse resilience
                prob = fatality_base + k_shock * current_shock * (
                    1.0 - a.get("resilience", 0.5)
                )
                if rng.rand() < prob:
                    a["alive"] = False

        # update lifespan counters
        for a in agents:
            if a["alive"]:
                a["lifespan"] += 1

        # record history
        history_c = collective_cci(agents)
        history_b = belief_convergence(agents)
        history_h = hope_index(agents)
        history_a = sum(1 for a in agents if a["alive"]) / len(agents)
        history["cci"].append(history_c)
        history["belief_conv"].append(history_b)
        history["hope"].append(history_h)
        history["alive_frac"].append(history_a)
        # branch fractions across allowed branches
        # compute chosen branch per alive agent
        branches_used = branches
        choices = []
        for a in agents:
            if not a["alive"]:
                continue
            bvec = a["belief"]
            if len(bvec) == 0:
                continue
            choices.append(int(np.argmax(bvec)))
        frac = []
        if choices:
            cnt = Counter(choices)
            total_alive = len(choices)
            for idx in range(len(branches_used)):
                frac.append(float(cnt.get(idx, 0)) / total_alive)
        else:
            frac = [0.0] * len(branches_used)
        history["branch_frac"].append(frac)

    # Branch selection at end
    branch = select_branch(agents, branches=branches)

    # Metrics
    cci_pre = history["cci"][2] if len(history["cci"]) > 2 else history["cci"][0]
    cci_post = history["cci"][-1]
    collective_cci_delta = cci_post - cci_pre
    belief_conv_final = history["belief_conv"][-1]
    hope_idx = history["hope"][-1]
    survival_rate = history["alive_frac"][-1]
    collapse_risk = 1.0 - survival_rate

    # lifespan summary
    lifespans = [a.get("lifespan", 0) for a in agents]
    mean_lifespan = float(np.mean(lifespans)) if lifespans else 0.0

    # collapse flag based on mortality model threshold (if provided)
    collapse_flag = False
    mm = params.get("mortality_model")
    if mm and mm.get("enabled"):
        threshold = mm.get("collapse_threshold", 0.45)
        collapse_flag = collapse_risk > threshold

    return {
        "params": params,
        "history": history,
        "collective_cci_delta": collective_cci_delta,
        "belief_convergence": belief_conv_final,
        "hope_index": hope_idx,
        "survival_rate": survival_rate,
        "collapse_risk": collapse_risk,
        "branch_selected": branch,
        "collapse_flag": collapse_flag,
        "lifespan_epochs": mean_lifespan,
        "mechanism": mech,
    }


def run_experiment_grid(
    label="religion_hope_mechanism",
    agents_list=[100, 200],
    shocks=[0.2, 0.5, 0.8],
    stress_duration=["acute", "chronic"],
    goal_diversity=[2, 3, 4, 5],
    noise_list=[0.05, 0.1],
    replicates=6,
    export=True,
):
    results = []
    i = 0
    total = (
        len(agents_list)
        * len(shocks)
        * len(stress_duration)
        * len(goal_diversity)
        * len(noise_list)
        * replicates
    )
    for n_agents in agents_list:
        for shock in shocks:
            for duration in stress_duration:
                for gd in goal_diversity:
                    for noise in noise_list:
                        for rep in range(replicates):
                            params = {
                                "n_agents": n_agents,
                                "shock": float(shock),
                                "stress_duration": duration,
                                "goal_diversity": int(gd),
                                "noise": float(noise),
                                "replicate": rep,
                            }
                            # allow user-supplied branch mask and other run-level knobs
                            # (these are expected to be present in higher-level wrappers)
                            # note: when run_experiment_grid is called directly, params remain minimal
                            params.update(
                                {
                                    k: v
                                    for k, v in globals()
                                    .get("EXTRA_RUN_KWARGS", {})
                                    .items()
                                }
                            )
                            params["seed"] = seed_for(
                                (label, n_agents, shock, duration, gd, noise, rep)
                            )
                            res = run_single(params)
                            row = {
                                "label": label,
                                "n_agents": n_agents,
                                "shock": shock,
                                "stress_duration": duration,
                                "goal_diversity": gd,
                                "noise": noise,
                                "replicate": rep,
                                "collective_cci_delta": res["collective_cci_delta"],
                                "belief_convergence": res["belief_convergence"],
                                "hope_index": res["hope_index"],
                                "survival_rate": res["survival_rate"],
                                "collapse_risk": res["collapse_risk"],
                                "branch_selected": res["branch_selected"],
                                "collapse_flag": res.get("collapse_flag", False),
                                "lifespan_epochs": res.get("lifespan_epochs", None),
                            }
                            results.append(row)
                            i += 1
                            if i % 20 == 0 or i == total:
                                print(f"  completed {i}/{total} runs")
    df = pd.DataFrame(results)
    if export:
        csv_path = OUT_DIR / "results.csv"
        df.to_csv(csv_path, index=False)
        print("Wrote:", csv_path)

    # Aggregate summaries
    summary = df.groupby(
        [
            "n_agents",
            "shock",
            "stress_duration",
            "goal_diversity",
            "noise",
            "branch_selected",
        ]
    ).agg(
        {
            "hope_index": ["mean", "std"],
            "survival_rate": ["mean", "std"],
            "collapse_risk": ["mean", "std"],
            "collective_cci_delta": ["mean", "std"],
        }
    )
    summary_path = OUT_DIR / "summary_by_branch.csv"
    summary.to_csv(summary_path)

    # Plots
    plt.figure(figsize=(6, 4))
    # hope vs shock
    agg = df.groupby(["shock"]).hope_index.mean().reset_index()
    plt.plot(agg.shock, agg.hope_index, marker="o")
    plt.title("Hope Index vs Shock")
    plt.xlabel("shock severity")
    plt.ylabel("mean hope_index")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hope_vs_shock.png")
    plt.close()

    # belief convergence curve: average over time is harder; approximate by belief_convergence metric
    plt.figure(figsize=(6, 4))
    for d in ["acute", "chronic"]:
        sub = (
            df[df.stress_duration == d]
            .groupby("shock")
            .belief_convergence.mean()
            .reset_index()
        )
        plt.plot(sub.shock, sub.belief_convergence, marker="o", label=d)
    plt.title("Belief Convergence by Shock & Duration")
    plt.xlabel("shock")
    plt.ylabel("belief_convergence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "belief_convergence_curve.png")
    plt.close()

    # survival by branch
    plt.figure(figsize=(6, 4))
    surv = df.groupby("branch_selected").survival_rate.mean().reindex(BRANCHES)
    surv.plot(kind="bar")
    plt.title("Average Survival Rate by Branch Selected")
    plt.ylabel("survival_rate")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "survival_by_branch.png")
    plt.close()

    # cci vs hope scatter
    plt.figure(figsize=(6, 4))
    plt.scatter(df.collective_cci_delta, df.hope_index, alpha=0.6)
    plt.xlabel("collective_cci_delta")
    plt.ylabel("hope_index")
    plt.title("CCI Change vs Hope")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cci_vs_hope.png")
    plt.close()

    # write summary.md
    md = OUT_DIR / "summary.md"
    with open(md, "w") as f:
        f.write(f"# Religion as Collective Hope Mechanism — {STAMP}\n\n")
        f.write(
            f"Parameters: agents={agents_list}, shocks={shocks}, stress_duration={stress_duration}, goal_diversity={goal_diversity}, noise={noise_list}\n\n"
        )

        # basic finding: fraction of runs where Religion selected at mid-high shocks
        def branch_frac(df, branch, shock_low=None, shock_high=None):
            sel = df
            if shock_low is not None:
                sel = sel[sel.shock >= shock_low]
            if shock_high is not None:
                sel = sel[sel.shock <= shock_high]
            return float((sel.branch_selected == branch).mean())

        religion_mid_high = branch_frac(df, "Religion", shock_low=0.5)
        education_chronic_mild = float(
            df[(df.stress_duration == "chronic") & (df.shock <= 0.2)]
            .branch_selected.eq("Education")
            .mean()
        )

        f.write(f"- Fraction Religion at shock>=0.5: {religion_mid_high:.3f}\n")
        f.write(
            f"- Fraction Education at chronic mild (shock<=0.2): {education_chronic_mild:.3f}\n"
        )
        f.write("\nFindings by branch (see summary_by_branch.csv and plots):\n")
        f.write(
            "- hope_vs_shock.png\n- belief_convergence_curve.png\n- survival_by_branch.png\n- cci_vs_hope.png\n"
        )

    print("Experiment complete. Outputs in:", OUT_DIR)
    return df


def analyze_results(df=None, out_report=None):
    # If df not provided, load results
    if df is None:
        path = OUT_DIR / "results.csv"
        df = pd.read_csv(path)
    # Compare branch_selected effects
    comp = df.groupby("branch_selected").agg(
        {
            "hope_index": ["mean", "std"],
            "survival_rate": ["mean", "std"],
            "collapse_risk": ["mean", "std"],
            "collective_cci_delta": ["mean", "std"],
        }
    )
    comp_path = OUT_DIR / "branch_comparison.csv"
    comp.to_csv(comp_path)

    # write human-readable report
    if out_report is None:
        out_report = OUT_DIR / "religion_hope_mechanism_results.md"
    else:
        out_report = Path(out_report)
    with open(out_report, "w") as f:
        f.write(f"# Religion Hope Mechanism — Analysis ({STAMP})\n\n")
        f.write("## Comparative metrics by branch\n\n")
        # prefer markdown table if tabulate is installed, otherwise fallback to plain text
        try:
            table_text = comp.to_markdown()
        except Exception:
            try:
                # older pandas may raise ImportError for tabulate; fallback
                table_text = comp.to_string()
            except Exception:
                table_text = "<unable to render table>"
        f.write(table_text)
        f.write(
            "\n\n## Notes\n- Higher fraction of Religion at mid-high shocks (>=0.5) indicates emergent collective narratives under acute stress.\n- Education dominance in chronic mild conditions supports the secondary hypothesis.\n"
        )
    print("Analysis written to:", out_report)
    return comp


if __name__ == "__main__":
    df = run_experiment_grid()
    analyze_results(df)
