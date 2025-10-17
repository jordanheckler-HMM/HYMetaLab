# tools/run_civ_regression.py
import os
import subprocess
import sys


def main():
    # Ensure output dir exists
    outdir = os.path.join("discovery_results", "civ_regression")
    os.makedirs(outdir, exist_ok=True)

    # Run the experiment
    cmd = [sys.executable, os.path.join("experiments", "civ_regression_runner.py")]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)

    # Print where to find files
    print("\n[Done] Exported files:")
    for root, _, files in os.walk(outdir):
        for fn in files:
            print(" -", os.path.join(root, fn))


if __name__ == "__main__":
    main()
