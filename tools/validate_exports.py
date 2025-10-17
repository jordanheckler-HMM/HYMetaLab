import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLS = {"CCI", "hazard", "risk", "survival", "seed", "epoch"}


def main():
    base = Path("results/discovery_results")
    ok = True
    for csv in base.rglob("*.csv"):
        try:
            df = pd.read_csv(csv)
            missing = REQUIRED_COLS - set(df.columns)
            if missing:
                print(f"[FAIL] {csv} missing {missing}")
                ok = False
        except Exception as e:
            print(f"[ERR] {csv}: {e}")
            ok = False
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
