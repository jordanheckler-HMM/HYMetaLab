#!/usr/bin/env python3
"""
Run follow-ups for open_min_budget:
1) reservoir fraction sweep for failing chemostat variants
2) agent-I/O equivalents with period=40 for multiple eps
3) increased seeds (n=6) for passing conditions
4) publish selected outputs into 04_LATEST_RESULTS and create root summaries

This script imports open_min_budget and uses its run_condition/analyze_and_export
functions while overriding module-level globals to run targeted sweeps.
"""
import json
import shutil
import time
from pathlib import Path

import open_min_budget as ob

ROOT = Path(".")
OUTROOT = Path("outputs/open_min_budget_followups")
OUTROOT.mkdir(parents=True, exist_ok=True)


def run_sweep(name, plan, defaults_override=None, seeds=None):
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    out = OUTROOT / f"{name}_{ts}"
    ob.OUT = out
    for d in ["data", "figures", "report", "bundle", "logs"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    # reset module accumulators
    ob.events = []
    ob.runs = []
    ob.traj = []
    # apply overrides
    if defaults_override:
        for k, v in defaults_override.items():
            ob.DEFAULTS[k] = v
    if seeds is not None:
        ob.DEFAULTS["seeds"] = seeds
    ob.PLAN = plan
    # run each condition
    for c in ob.PLAN:
        ob.run_condition(c)
    ob.analyze_and_export()
    return out


def publish_latest(
    outputs_dirs, dest_dir=Path("04_LATEST_RESULTS/open_min_budget_followups_latest")
):
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True)
    summary = []
    for p in outputs_dirs:
        # copy entire folder
        name = p.name
        shutil.copytree(p, dest_dir / name)
        # collect summary
        with open(p / "data" / "open_min_budget_summary.json") as f:
            s = json.load(f)
        summary.append(
            {"out": str(p), "winner": s.get("winner"), "timestamp": s.get("timestamp")}
        )
    # write root-level summary files
    RUNS_CSV = Path("OPEN_MIN_BUDGET_FOLLOWUPS_RUNS.csv")
    with open(RUNS_CSV, "w") as f:
        f.write("out_path,winner,timestamp\n")
        for s in summary:
            f.write(f"{s['out']},{s['winner']},{s['timestamp']}\n")
    with open("OPEN_MIN_BUDGET_FOLLOWUPS_SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)
    # simple markdown report
    with open("OPEN_MIN_BUDGET_FOLLOWUPS_REPORT.md", "w") as f:
        f.write("# open_min_budget followups summary\n\n")
        for s in summary:
            f.write(f"- {s['out']} winner={s['winner']} timestamp={s['timestamp']}\n")
    return summary


def main():
    outs = []

    # 1) reservoir sweep for failing chemostat variants
    chems = [
        {
            "label": "chemostat_eps0.0020_period40",
            "mech": "chemostat",
            "epsilon": 0.0020,
            "period": 40,
        },
        {
            "label": "chemostatCTRL_eps0.0025_period40",
            "mech": "chemostat_ctrl",
            "epsilon": 0.0025,
            "period": 40,
        },
    ]
    for frac in [0.25, 0.30, 0.35]:
        name = f"reservoir_frac_{int(frac*100)}"
        defaults_override = {"reservoir_energy_frac": frac}
        out = run_sweep(name, chems, defaults_override=defaults_override)
        outs.append(out)

    # 2) agent-I/O equivalents with period=40 for eps 0.0015,0.0020,0.0025
    agent_plan = []
    for eps in [0.0015, 0.0020, 0.0025]:
        agent_plan.append(
            {
                "label": f"agentio_eps{eps}_period40",
                "mech": "agent_io",
                "epsilon": eps,
                "period": 40,
            }
        )
    out = run_sweep("agentio_period40", agent_plan)
    outs.append(out)

    # 3) increase seeds for passing conditions (closed, agentio_eps0.0025, chemostat_eps0.0015)
    seeds6 = [101, 202, 303, 404, 505, 606]
    pass_plan = [
        {"label": "closed", "mech": "none", "epsilon": 0.0, "period": None},
        {
            "label": "agentio_eps0.0025",
            "mech": "agent_io",
            "epsilon": 0.0025,
            "period": 20,
        },
        {
            "label": "chemostat_eps0.0015_period40",
            "mech": "chemostat",
            "epsilon": 0.0015,
            "period": 40,
        },
    ]
    out = run_sweep("passing_conditions_n6", pass_plan, seeds=seeds6)
    outs.append(out)

    # 4) publish selected outputs
    summary = publish_latest(outs)
    print(
        "Followups completed; published to 04_LATEST_RESULTS/open_min_budget_followups_latest"
    )
    print(
        "Root-level summary files: OPEN_MIN_BUDGET_FOLLOWUPS_RUNS.csv, OPEN_MIN_BUDGET_FOLLOWUPS_SUMMARY.json, OPEN_MIN_BUDGET_FOLLOWUPS_REPORT.md"
    )


if __name__ == "__main__":
    main()
