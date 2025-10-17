# tools/run_agent_strategies.py
import os
import subprocess
import sys


def main():
    outdir = os.path.join("discovery_results", "agent_strategies")
    os.makedirs(outdir, exist_ok=True)
    cmd = [sys.executable, os.path.join("experiments", "agent_strategies_runner.py")]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(rc)
    print("\n[Done] Exported files:")
    for root, _, files in os.walk(outdir):
        for fn in files:
            print(" -", os.path.join(root, fn))


if __name__ == "__main__":
    main()
