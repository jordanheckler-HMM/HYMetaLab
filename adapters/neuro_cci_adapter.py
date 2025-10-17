from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Minimal processing utilities — replace with your project's signal-processing where available
def compute_cal(eeg_epoch: np.ndarray) -> float:
    # calibration: normalized mean power in band
    return float(np.mean(np.abs(eeg_epoch)))


def compute_coh(eeg_epoch: np.ndarray) -> float:
    # coherence proxy: inverse of variance across channels
    try:
        if eeg_epoch.ndim == 1:
            return 1.0
        return float(1.0 / (1e-6 + np.var(eeg_epoch, axis=0).mean()))
    except Exception:
        return 1.0


def compute_emotion_likelihoods(
    features: dict[str, float], priors: dict[str, dict[str, float]]
):
    # Simple Bayesian score: product of likelihoods weighted by priors
    scores = {}
    for emo, params in priors.items():
        s = params.get("prior", 0.01)
        for k, v in params.get("features", {}).items():
            mu = v.get("mu", 0.0)
            sig = v.get("sigma", 1.0)
            x = features.get(k, 0.0)
            like = np.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            s *= like
        scores[emo] = s
    total = sum(scores.values()) + 1e-12
    for k in scores:
        scores[k] = float(scores[k] / total)
    return scores


def load_priors(path: Path):
    if not path.exists():
        return {
            "Love": {"prior": 0.3, "features": {"cal": {"mu": 0.6, "sigma": 0.2}}},
            "Anger": {"prior": 0.2, "features": {"cal": {"mu": 0.4, "sigma": 0.2}}},
            "Sadness": {"prior": 0.1, "features": {"cal": {"mu": 0.2, "sigma": 0.2}}},
        }
    return json.loads(open(path).read())


def run_study(config: dict[str, Any], out_dir: str, seed: int):
    rng = np.random.default_rng(seed)
    cfg = config
    base = Path(out_dir)
    (base / "data").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "report").mkdir(parents=True, exist_ok=True)

    input_glob = cfg.get("constants", {}).get("input_glob", "data/input/neuro/*.csv")
    eps = cfg.get("constants", {}).get("epsilon", 1e-3)
    priors_path = Path("data/priors/Meaning_Table_v0.2.json")
    priors = load_priors(priors_path)

    rows = []
    trajs = []

    import glob

    files = glob.glob(input_glob)
    for p in files:
        df = pd.read_csv(p)
        if "participant" not in df.columns or "block" not in df.columns:
            continue
        eeg_cols = [c for c in df.columns if c.startswith("eeg_")]
        for _, row in df.iterrows():
            pid = row.get("participant")
            blk = row.get("block")
            epoch = int(row.get("epoch", 0))
            eeg = row[eeg_cols].to_numpy(dtype=float)
            cal = compute_cal(eeg)
            coh = compute_coh(eeg)
            em = float(row.get("behavior", 0.0))
            noise = float(row.get("hrv", 0.0)) + 1e-6
            cci = (cal * coh * em) / (eps + noise)

            features = {"cal": cal, "coh": coh, "em": em, "noise": noise}
            emo_scores = compute_emotion_likelihoods(features, priors)

            run_id = f"p{pid}_b{blk}_s{seed}"
            rec = {
                "run_id": run_id,
                "seed": seed,
                "participant": pid,
                "block": blk,
                "epoch": epoch,
                "Cal": cal,
                "Coh": coh,
                "Em": em,
                "Noise": noise,
                "CCI": cci,
            }
            for k, v in emo_scores.items():
                rec[f"emo_{k}"] = v
            rows.append(rec)
            trajs.append(
                {
                    "run_id": run_id,
                    "seed": seed,
                    "epoch": epoch,
                    "CCI": cci,
                    "Cal": cal,
                    "Coh": coh,
                    "Em": em,
                    "Noise": noise,
                }
            )

    if not rows:
        raise RuntimeError(
            "No input data found. Place participant CSVs matching the input_glob."
        )

    pd.DataFrame(rows).to_csv(base / "data" / "runs_summary.csv", index=False)
    pd.DataFrame(trajs).to_csv(base / "data" / "trajectories_long.csv", index=False)

    trajs_df = pd.DataFrame(trajs)
    plt.figure(figsize=(8, 4))
    for rid, g in trajs_df.groupby("run_id"):
        plt.plot(g["epoch"], g["CCI"], alpha=0.4)
    plt.title("CCI trajectories")
    plt.xlabel("epoch")
    plt.ylabel("CCI")
    plt.savefig(base / "figures" / "cci_long.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.scatter(trajs_df["Cal"], trajs_df["Coh"], s=6)
    plt.xlabel("Cal")
    plt.ylabel("Coh")
    plt.title("Cal vs Coh (proxy for PLV)")
    plt.savefig(base / "figures" / "coh_plv.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 3))
    h = trajs_df.groupby("epoch")["Noise"].mean().reset_index()
    plt.plot(h["epoch"], h["Noise"])
    plt.title("HRV (Noise) over time")
    plt.xlabel("epoch")
    plt.ylabel("HRV")
    plt.savefig(base / "figures" / "hrv_time.png", dpi=200)
    plt.close()

    rep = base / "report" / "neuro_cci_results.md"
    rep.write_text("# Neuro CCI Results\n\nSee figures in ../figures/\n")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bundle_path = base / "bundle"
    bundle_path.mkdir(exist_ok=True)
    zipname = bundle_path / f"neuro_cci_{stamp}.zip"
    with zipfile.ZipFile(zipname, "w") as z:
        z.write(base / "data" / "runs_summary.csv", arcname="runs_summary.csv")
        z.write(
            base / "data" / "trajectories_long.csv", arcname="trajectories_long.csv"
        )
        # add figures
        for f in (base / "figures").glob("*"):
            z.write(f, arcname=f"figures/{f.name}")
        # add report
        z.write(rep, arcname=f"report/{rep.name}")

    return {
        "runs_summary_csv": str(base / "data" / "runs_summary.csv"),
        "trajectories_long_csv": str(base / "data" / "trajectories_long.csv"),
        "summary_zip": str(zipname),
    }
