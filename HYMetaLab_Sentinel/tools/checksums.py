import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_payload(payload: dict) -> str:
    b = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def write_tree_sums(root: str, out: str):
    rootp = Path(root)
    lines = []
    for fp in rootp.rglob("*"):
        if fp.is_file():
            lines.append(f"{sha256_file(fp)} {fp}")
    Path(out).write_text("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    Path("dist").mkdir(exist_ok=True)
    print(write_tree_sums("dist", "dist/SHA256SUMS.txt"))
