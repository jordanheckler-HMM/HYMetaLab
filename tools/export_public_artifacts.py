"""Copy summary files, figures, and report from a discovery run to public_artifacts/ for safe publishing.

Usage:
  python -m tools.export_public_artifacts <path_to_run_dir>

This will copy `data/runs_summary.csv`, `data/summary_bootstrap.json`, any files under `figures/` and `report/` into
`public_artifacts/<run_name>/` so you can commit that smaller bundle to the repo without exposing raw trajectories.
"""

import shutil
import sys
from pathlib import Path


def export(run_dir: Path, out_base: Path = Path("public_artifacts")):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir {run_dir} not found")
    name = run_dir.name
    dst = out_base / name
    if dst.exists():
        print(f"Overwriting existing {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)

    to_copy = []
    candidates = [
        run_dir / "data" / "runs_summary.csv",
        run_dir / "data" / "summary_bootstrap.json",
    ]
    for c in candidates:
        if c.exists():
            to_copy.append(c)

    for sub in ["figures", "report"]:
        p = run_dir / sub
        if p.exists():
            for f in p.glob("*"):
                if f.is_file():
                    to_copy.append(f)

    if not to_copy:
        print("No public artifacts found to export.")
        return

    for f in to_copy:
        rel = f.relative_to(run_dir)
        dstf = dst / rel
        dstf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dstf)
        print(f"Copied {f} -> {dstf}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.export_public_artifacts <run_dir>")
        raise SystemExit(1)
    export(Path(sys.argv[1]))
