#!/usr/bin/env python3
"""
Add Bootstrap Confidence Intervals to Study Summaries
Idempotent - safe to run multiple times
"""
import csv
import glob
import json
import random
import statistics as st
from pathlib import Path


def boot_ci(vals, B=1200, alpha=0.05):
    """Compute bootstrap CI for a list of values"""
    if not vals or len(vals) < 2:
        return None, None, None
    n = len(vals)
    bs = [st.mean(random.choices(vals, k=n)) for _ in range(B)]
    lo = int(B * alpha / 2)
    hi = int(B * (1 - alpha / 2))
    bs.sort()
    return (st.mean(vals), bs[lo], bs[hi])


def read_metrics_from_csv(path):
    """Extract delta_cci and delta_hazard from CSV"""
    dcci, dhaz = [], []
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Try to find delta_cci column (various names)
                for k in row.keys():
                    lk = k.lower()
                    if "delta" in lk and "cci" in lk:
                        try:
                            dcci.append(float(row[k]))
                            break
                        except:
                            pass

                # Try to find delta_hazard column
                for k in row.keys():
                    lk = k.lower()
                    if ("hazard" in lk or "dhazard" in lk) and "delta" in lk:
                        try:
                            dhaz.append(float(row[k]))
                            break
                        except:
                            pass
    except Exception as e:
        print(f"⚠️  Error reading {path}: {e}")

    return dcci, dhaz


def inject_ci_into_summary(summary_path, dcci_ci, dhaz_ci):
    """Add CI data to summary.json and update classification"""
    # Load existing summary or create new
    if summary_path.exists():
        try:
            J = json.loads(summary_path.read_text())
        except:
            J = {}
    else:
        J = {}

    # Ensure metrics dict exists
    J.setdefault("metrics", {})

    # Add CCI metrics
    if dcci_ci[0] is not None:
        J["metrics"]["delta_cci_mean"] = round(dcci_ci[0], 6)
        J["metrics"]["delta_cci_ci_lo"] = round(dcci_ci[1], 6)
        J["metrics"]["delta_cci_ci_hi"] = round(dcci_ci[2], 6)

    # Add hazard metrics
    if dhaz_ci[0] is not None:
        J["metrics"]["delta_hazard_mean"] = round(dhaz_ci[0], 6)
        J["metrics"]["delta_hazard_ci_lo"] = round(dhaz_ci[1], 6)
        J["metrics"]["delta_hazard_ci_hi"] = round(dhaz_ci[2], 6)

    # Update classification based on CI (conservative: use CI bounds)
    def valid(x):
        return x is not None and not (
            isinstance(x, float) and (x != x)
        )  # Check for NaN

    cci_validated = (
        valid(J["metrics"].get("delta_cci_ci_lo"))
        and J["metrics"]["delta_cci_ci_lo"] > 0.03
    )
    haz_validated = (
        valid(J["metrics"].get("delta_hazard_ci_hi"))
        and J["metrics"]["delta_hazard_ci_hi"] < -0.01
    )

    if cci_validated and haz_validated:
        cls = "VALIDATED"
    elif cci_validated or haz_validated:
        cls = "PARTIAL"
    else:
        cls = J.get("classification", {}).get("status", "UNDER_REVIEW")

    J["classification"] = {"status": cls}

    # Write back
    summary_path.write_text(json.dumps(J, indent=2))
    return cls


def main():
    """Process all discovery results directories"""
    # Find all potential result directories
    patterns = [
        "discovery_results/*/*/",
        "discovery_results/*/",
    ]

    dirs = set()
    for pattern in patterns:
        for path in glob.glob(pattern):
            if Path(path).is_dir():
                dirs.add(Path(path))

    touched = 0
    validated = 0
    partial = 0

    for result_dir in sorted(dirs):
        # Look for CSV files with results
        csv_files = (
            list(result_dir.glob("runs.csv"))
            + list(result_dir.glob("*_results.csv"))
            + list(result_dir.glob("*_results_seed*.csv"))
        )

        if not csv_files:
            continue

        # Aggregate metrics across all CSVs
        all_dcci, all_dhaz = [], []
        for csv_file in csv_files:
            dcci, dhaz = read_metrics_from_csv(csv_file)
            all_dcci.extend(dcci)
            all_dhaz.extend(dhaz)

        if not all_dcci and not all_dhaz:
            continue

        # Compute bootstrap CIs
        dcci_ci = boot_ci(all_dcci) if all_dcci else (None, None, None)
        dhaz_ci = boot_ci(all_dhaz) if all_dhaz else (None, None, None)

        # Update summary
        summary_path = result_dir / "summary.json"
        cls = inject_ci_into_summary(summary_path, dcci_ci, dhaz_ci)

        print(f"✅ {result_dir.name}")
        if dcci_ci[0] is not None:
            print(f"   ΔCCI: {dcci_ci[0]:.4f} [{dcci_ci[1]:.4f}, {dcci_ci[2]:.4f}]")
        if dhaz_ci[0] is not None:
            print(f"   Δhazard: {dhaz_ci[0]:.4f} [{dhaz_ci[1]:.4f}, {dhaz_ci[2]:.4f}]")
        print(f"   Classification: {cls}")

        touched += 1
        if cls == "VALIDATED":
            validated += 1
        elif cls == "PARTIAL":
            partial += 1

    print("\n✅ Bootstrap CI updates complete")
    print(f"   Directories processed: {touched}")
    print(f"   VALIDATED: {validated}")
    print(f"   PARTIAL: {partial}")


if __name__ == "__main__":
    main()
