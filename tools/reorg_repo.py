import glob
import hashlib
import json
import os
import shutil
import time
import zipfile

BASE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(BASE, ".."))


def ensure_dirs():
    for p in [
        "studies",
        "experiments_exploratory",
        "adapters",
        "sim",
        "analysis",
        "results/discovery_results",
        "results/archive",
        "docs/whitepaper",
        "docs/meaning",
        "docs/integrity",
    ]:
        os.makedirs(os.path.join(ROOT, p), exist_ok=True)


def move_if_exists(patterns, dest):
    for pat in patterns:
        for f in glob.glob(os.path.join(ROOT, pat)):
            d = os.path.join(ROOT, dest, os.path.basename(f))
            if os.path.isdir(f):
                shutil.move(f, os.path.join(ROOT, dest))
            else:
                shutil.move(f, d)


def checksum_and_archive(run_dir):
    if not os.path.isdir(run_dir):
        return None
    zpath = os.path.join(ROOT, "results/archive", f"{os.path.basename(run_dir)}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(run_dir):
            for fn in files:
                fp = os.path.join(root, fn)
                z.write(fp, os.path.relpath(fp, run_dir))
    # sha256
    h = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
    meta = {"archive": zpath, "sha256": h, "created": time.time()}
    with open(zpath + ".sha256.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ensure_dirs()
    # move docs (keep your existing file names)
    move_if_exists(["*Whitepaper*.md", "*Atlas*.pdf"], "docs/whitepaper")
    move_if_exists(["Meaning*Periodic*Table*.pdf"], "docs/meaning")
    move_if_exists(
        [
            "*Case*Studies*Prompt*.pdf",
            "*OpenLaws*.pdf",
            "*Integrity*.md",
            "*Project_Log*.pdf",
        ],
        "docs/integrity",
    )
    # optional: archive any legacy run dirs into archive with checksum
    runs = glob.glob(os.path.join(ROOT, "discovery_results", "*")) + glob.glob(
        os.path.join(ROOT, "results", "discovery_results", "*")
    )
    for r in runs:
        meta = checksum_and_archive(r)
        if meta:
            print("Archived:", meta["archive"], meta["sha256"])


if __name__ == "__main__":
    main()
