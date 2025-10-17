import os
import sys

# ensure project root is on path so 'sim' package can be imported when running this script directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import json
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import yaml

from sim.agent import Agent
from sim.io_utils import write_jsonl, write_run_manifest


def run_calibration(params=None, outdir=None, config_path=None):
    # Handle parameter dict if passed
    if isinstance(params, dict):
        # Extract parameters from dict
        seeds = params.get("seeds", [42])
        noise_levels = params.get("noise_levels", [0.1])
        n_agents = params.get("n_agents", 20)
        ticks = params.get("ticks", 20)
        export_base = params.get("export_base", "outputs/calibration")
        bins = params.get("bins", 10)
        bootstrap_samples = params.get("bootstrap_samples", 200)
    else:
        # Fallback to config file if provided
        if config_path and os.path.exists(config_path):
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            seeds = cfg.get("seeds", [42])
            noise_levels = cfg.get("noise_levels", [0.1])
            n_agents = cfg.get("n_agents", 20)
            ticks = cfg.get("ticks", 20)
            export_base = cfg.get("export_base", "outputs/calibration")
            bins = cfg.get("bins", 10)
            bootstrap_samples = cfg.get("bootstrap_samples", 200)
        else:
            # Default values
            seeds = [42]
            noise_levels = [0.1]
            n_agents = 20
            ticks = 20
            export_base = "outputs/calibration"
            bins = 10
            bootstrap_samples = 200

    # Use outdir if provided
    if outdir:
        export_base = outdir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(export_base, f"calibration_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    all_results = {}
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        for noise in noise_levels:
            records = []
            # simple synthetic decisions: agent reports confidence ~ true probability + noise
            for t in range(ticks):
                for i in range(n_agents):
                    a = Agent(agent_id=i)
                    # true success prob: depends on agent and tick (make some structure)
                    true_p = 0.3 + 0.4 * np.sin((i + t) * 0.1)
                    true_p = float(np.clip(true_p, 0.01, 0.99))
                    outcome = 1 if random.random() < true_p else 0
                    reported = float(
                        np.clip(true_p + np.random.normal(0, noise), 0.0, 1.0)
                    )
                    rec = {
                        "tick": t,
                        "agent_id": f"A-{i}",
                        "chosen_action": "test",
                        "reported_conf": reported,
                        "outcome_reward": outcome,
                        "true_p": true_p,
                    }
                    records.append(rec)

            subdir = os.path.join(run_dir, f"seed_{seed}_noise_{int(noise*1000)}")
            os.makedirs(subdir, exist_ok=True)
            decisions_path = os.path.join(subdir, "decisions.jsonl")
            write_jsonl(decisions_path, records)
            write_run_manifest(subdir, {"seed": seed, "noise": noise}, seed)

            # compute calibration by bin
            df_bins = defaultdict(list)
            for r in records:
                b = int(r["reported_conf"] * bins)
                if b == bins:
                    b = bins - 1
                df_bins[b].append(r)

            calib = []
            for b in range(bins):
                recs = df_bins.get(b, [])
                if not recs:
                    calib.append(
                        {"bin": b, "n": 0, "avg_reported": None, "empirical": None}
                    )
                    continue
                avg_rep = float(np.mean([x["reported_conf"] for x in recs]))
                emp = float(np.mean([x["outcome_reward"] for x in recs]))
                calib.append(
                    {
                        "bin": b,
                        "n": len(recs),
                        "avg_reported": avg_rep,
                        "empirical": emp,
                    }
                )

            # compute bootstrap CIs for empirical frequency in each bin
            bs_samples = bootstrap_samples
            for c in calib:
                if c["n"] == 0:
                    c["ci_lower"] = None
                    c["ci_upper"] = None
                    continue
                outcomes = [x["outcome_reward"] for x in df_bins[c["bin"]]]
                boots = []
                for _ in range(bs_samples):
                    resample = np.random.choice(
                        outcomes, size=len(outcomes), replace=True
                    )
                    boots.append(float(np.mean(resample)))
                ci_lower = float(np.percentile(boots, 2.5))
                ci_upper = float(np.percentile(boots, 97.5))
                c["ci_lower"] = ci_lower
                c["ci_upper"] = ci_upper

            summary_path = os.path.join(subdir, "calibration_summary.json")
            with open(summary_path, "w") as sf:
                json.dump({"calibration": calib}, sf, indent=2)

            all_results[f"seed_{seed}_noise_{noise}"] = {
                "decisions": decisions_path,
                "summary": summary_path,
            }

    return {"run_dir": run_dir, "results": all_results}


if __name__ == "__main__":
    out = run_calibration("configs/calibration.yaml")
    print(out)
