"""Generate a small run manifest for public release bundles.

Writes `run_manifest.json` containing study id, timestamp, seeds, sweep params, python version,
requirements checksum, and a small hash of the study YAML and adapter file.
"""

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def file_sha(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(8192)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def generate_manifest(
    run_dir: str, study_yaml: str = None, adapter_path: str = None, extra: dict = None
):
    ROOT = Path(".").resolve()
    run_dir = Path(run_dir)
    out = run_dir / "run_manifest.json"
    manifest = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "python": sys.version.replace("\n", " "),
        "requirements_sha256": None,
        "study_yaml_sha256": None,
        "adapter_sha256": None,
        "extra": extra or {},
    }

    req = ROOT / "requirements.txt"
    if req.exists():
        manifest["requirements_sha256"] = file_sha(req)
    if study_yaml:
        sy = Path(study_yaml)
        if sy.exists():
            manifest["study_yaml_sha256"] = file_sha(sy)
    if adapter_path:
        ap = Path(adapter_path)
        if ap.exists():
            manifest["adapter_sha256"] = file_sha(ap)

    # If study YAML provided, try to extract seeds and sweep params for reproducibility
    if study_yaml:
        sy = Path(study_yaml)
        if sy.exists():
            try:
                import yaml

                data = yaml.safe_load(open(sy))
                seeds = None
                sweep = None
                if isinstance(data, dict):
                    seeds = data.get("seed_policy", {}).get("seeds") or data.get(
                        "seed_policy"
                    )
                    sweep = data.get("sweep")
                manifest["study_seeds"] = seeds
                manifest["study_sweep"] = sweep
            except Exception:
                pass

    # Git commit info
    try:
        import subprocess

        git_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT))
            .decode()
            .strip()
        )
        manifest["git_commit"] = git_sha
    except Exception:
        manifest["git_commit"] = None

    out.write_text(json.dumps(manifest, indent=2))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--study_yaml", default=None)
    ap.add_argument("--adapter", default=None)
    args = ap.parse_args()
    p = generate_manifest(
        args.run_dir, study_yaml=args.study_yaml, adapter_path=args.adapter
    )
    print("Wrote", p)
