import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from openlaws_studies.evals.metrics import (
    measure_calibration,
    measure_coherence,
    measure_emergence,
    measure_noise,
)


@dataclass
class Config:
    seeds: list[int]
    bootstrap_n: int
    confidence: float
    out_dir: str


def bootstrap_ci(x: np.ndarray, n_boot: int = 800, ci: float = 0.95):
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(x, size=len(x), replace=True)
        boots.append(sample.mean())
    boots = np.array(boots)
    low = np.percentile(boots, (1 - ci) / 2 * 100)
    high = np.percentile(boots, (1 + (ci)) / 2 * 100)
    return float(x.mean()), float(low), float(high)


def run_adapter(config: Config) -> dict:
    os.makedirs(config.out_dir, exist_ok=True)
    rows = []
    for seed in config.seeds:
        Cal = measure_calibration(seed)
        Coh = measure_coherence(seed)
        Em = measure_emergence(seed)
        Noise = measure_noise(seed)
        cci_raw = (Cal * Coh * Em) / max(Noise, 1e-6)
        cci_norm = cci_raw / (1.0 + cci_raw)
        rows.append(
            {
                "seed": seed,
                "Cal": Cal,
                "Coh": Coh,
                "Em": Em,
                "Noise": Noise,
                "CCI_raw": cci_raw,
                "CCI_norm": cci_norm,
            }
        )

    df = pd.DataFrame(rows).sort_values("seed")
    df.to_csv(os.path.join(config.out_dir, "runs_summary.csv"), index=False)

    mean, lo, hi = bootstrap_ci(
        df["CCI_norm"].to_numpy(), n_boot=config.bootstrap_n, ci=config.confidence
    )
    sigma = float(df["CCI_norm"].std(ddof=1))
    summary = {
        "CCI_mean": mean,
        "CI_lower": lo,
        "CI_upper": hi,
        "sigma_run": sigma,
        "n_runs": len(df),
    }
    with open(os.path.join(config.out_dir, "bootstrap_ci.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
