import csv
import json
import os
import shutil

REQUIRED_KEYS = ["cci_mean", "hazard_mean", "survival_mean"]
BOUNDS = {
    "cci_mean": (0.0, 1.0),
    "hazard_mean": (0.0, 1.0),
    "survival_mean": (0.0, 1.0),
}


def _within(v, lo, hi):
    try:
        return (v is not None) and (lo <= float(v) <= hi)
    except:
        return False


def validate_metrics_json(path):
    with open(path) as f:
        data = json.load(f)
    for k in REQUIRED_KEYS:
        if k not in data:
            return False, f"Missing {k}"
        lo, hi = BOUNDS[k]
        if not _within(data[k], lo, hi):
            return False, f"{k} out of bounds: {data[k]}"
    return True, "ok"


def validate_csv_nonempty(path, min_rows=2):
    with open(path) as f:
        r = list(csv.reader(f))
    return (len(r) >= min_rows), f"rows={len(r)}"


def quarantine(folder, reason):
    qdir = os.path.join(os.path.dirname(folder), "_quarantine")
    os.makedirs(qdir, exist_ok=True)
    dest = os.path.join(qdir, os.path.basename(folder))
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(folder, dest)
    with open(os.path.join(qdir, "README.txt"), "a") as log:
        log.write(f"{os.path.basename(folder)} -> {reason}\n")
    return dest


def hazmat_sweep(export_dir):
    if not os.path.isdir(export_dir):
        return []
    acts = []
    for name in os.listdir(export_dir):
        run_dir = os.path.join(export_dir, name)
        if not os.path.isdir(run_dir):
            continue
        mj = os.path.join(run_dir, "metrics.json")
        sc = os.path.join(run_dir, "summary.csv")
        ok1, msg1 = (False, "no metrics.json")
        if os.path.isfile(mj):
            ok1, msg1 = validate_metrics_json(mj)
        ok2, msg2 = (False, "no summary.csv")
        if os.path.isfile(sc):
            ok2, msg2 = validate_csv_nonempty(sc)
        if not (ok1 and ok2):
            dest = quarantine(run_dir, f"metrics:{msg1} csv:{msg2}")
            acts.append((name, "quarantined", dest))
        else:
            acts.append((name, "ok", run_dir))
    return acts
