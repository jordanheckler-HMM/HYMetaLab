import json
import os
import random
from datetime import datetime

import numpy as np

try:
    from lifelines import KaplanMeierFitter

    HAVE_LIFELINES = True
except Exception:
    HAVE_LIFELINES = False


def run_survival(
    params=None,
    outdir=None,
    n_subjects=100,
    max_time=100,
    treatment_effect=0.2,
    seed=42,
    export_base="outputs/survival",
):
    # Handle parameter dict if passed
    if isinstance(params, dict):
        n_subjects = params.get("n_subjects", n_subjects)
        max_time = params.get("max_time", max_time)
        treatment_effect = params.get("treatment_effect", treatment_effect)
        seed = params.get("seed", seed)
        export_base = params.get("export_base", export_base)

    # Use outdir if provided
    if outdir:
        export_base = outdir

    random.seed(seed)
    np.random.seed(seed)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(export_base, f"survival_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # simulate survival times: exponential baseline with treatment reducing hazard
    subjects = []
    for i in range(n_subjects):
        treated = i < n_subjects // 2
        hazard = 0.05 * (1.0 - treatment_effect) if treated else 0.05
        # draw survival time from exponential
        time = np.random.exponential(1.0 / hazard)
        censored = time > max_time
        obs_time = min(time, max_time)
        subjects.append(
            {
                "id": i,
                "treated": treated,
                "time": float(obs_time),
                "event": 0 if censored else 1,
            }
        )

    summary_path = os.path.join(run_dir, "survival_subjects.json")
    with open(summary_path, "w") as f:
        json.dump({"subjects": subjects}, f, indent=2)

    if HAVE_LIFELINES:
        import matplotlib.pyplot as plt

        kmf_t = KaplanMeierFitter()
        kmf_c = KaplanMeierFitter()
        treated_times = [s["time"] for s in subjects if s["treated"]]
        treated_events = [s["event"] for s in subjects if s["treated"]]
        control_times = [s["time"] for s in subjects if not s["treated"]]
        control_events = [s["event"] for s in subjects if not s["treated"]]
        plt.figure()
        kmf_t.fit(treated_times, event_observed=treated_events, label="treated")
        ax = kmf_t.plot_survival_function()
        kmf_c.fit(control_times, event_observed=control_events, label="control")
        kmf_c.plot_survival_function(ax=ax)
        out_png = os.path.join(run_dir, "km_plot.png")
        plt.savefig(out_png)
        return {"run_dir": run_dir, "plot": out_png, "summary": summary_path}
    else:
        return {
            "run_dir": run_dir,
            "summary": summary_path,
            "note": "lifelines not installed; install lifelines for plots",
        }


if __name__ == "__main__":
    print(run_survival())
