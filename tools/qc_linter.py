import json
import pathlib
import re
import sys
from collections import defaultdict

import yaml

ROOT = pathlib.Path(".")
BANNED = [
    r"\bprove[sd]?\b",
    r"\bconfirm(?:s|ed)?\b",
    r"\bdefinitive\b",
    r"\bconclusive(?:ly)?\b",
    r"\buniversal law\b",
    r"\bapplies to all systems\b",
    r"\bempirical validation\b",
    r"\bbreakthrough\b",
    r"\bgroundbreaking\b",
    r"\brevolutionary\b",
    r"\bparadigm shift\b",
    r"\b(comparable|equivalent)\s+to\s+(thermodynamics|newton|einstein)\b",
]
OK_SIM_SCOPING = [
    r"within simulation context",
    r"simulation-validated",
    r"hypothesis requiring empirical testing",
]


def read_texts():
    for p in ROOT.rglob("*"):
        if p.suffix.lower() in {".md", ".txt"}:
            yield p, p.read_text(errors="ignore")
        if p.suffix.lower() in {".json", ".yml", ".yaml"}:
            yield p, p.read_text(errors="ignore")


def load_yaml_json(p):
    try:
        if p.suffix.lower() in {".yml", ".yaml"}:
            return yaml.safe_load(p.read_text())
        if p.suffix.lower() == ".json":
            return json.loads(p.read_text())
    except Exception:
        return None


def find_studies():
    studies = []
    for p in ROOT.rglob("studies/*.yml"):
        y = load_yaml_json(p)
        if isinstance(y, dict):
            studies.append((p, y))
    return studies


def banned_hits(text):
    hits = []
    for pat in BANNED:
        for m in re.finditer(pat, text, flags=re.I):
            hits.append((m.group(0), m.start()))
    return hits


def has_any(text, pats):
    return any(re.search(p, text, flags=re.I) for p in pats)


def main():
    report = []
    meta = {}
    for p in ROOT.rglob("**/summary.json"):
        j = load_yaml_json(p)
        if isinstance(j, dict) and "study_id" in j:
            meta[j["study_id"]] = {"path": str(p), "data": j}

    text_flags = []
    for p, text in read_texts():
        hits = banned_hits(text)
        if hits:
            text_flags.append({"file": str(p), "hits": hits})

    circ_graph = defaultdict(set)
    per_study = []
    for p, y in find_studies():
        sid = y.get("study_id", p.stem)
        ds = y.get("data_source") or meta.get(sid, {}).get("data", {}).get(
            "data_source"
        )
        thr = y.get("thresholds") or meta.get(sid, {}).get("data", {}).get("thresholds")
        deps = y.get("depends_on") or meta.get(sid, {}).get("data", {}).get(
            "depends_on", []
        )
        basis = y.get("independent_validation_basis") or meta.get(sid, {}).get(
            "data", {}
        ).get("independent_validation_basis")
        abstract_path = ROOT / p.parent / "abstract.md"
        abstract_txt = (
            abstract_path.read_text(errors="ignore") if abstract_path.exists() else ""
        )

        errs = []
        if ds not in {"SIMULATION_ONLY", "EMPIRICAL_PARTIAL", "EMPIRICAL_FULL"}:
            errs.append(
                "Missing/invalid data_source (SIMULATION_ONLY | EMPIRICAL_PARTIAL | EMPIRICAL_FULL)"
            )

        if ds == "SIMULATION_ONLY":
            if abstract_path.exists():
                if not has_any(abstract_txt, OK_SIM_SCOPING):
                    errs.append(
                        "SIMULATION_ONLY abstract missing scoping phrase (e.g., 'within simulation context')"
                    )
                if any(
                    "empirical validation" in h[0].lower()
                    for h in banned_hits(abstract_txt)
                ):
                    errs.append(
                        "SIMULATION_ONLY uses 'empirical validation' in abstract"
                    )
            else:
                errs.append("Missing abstract.md (required to host scoping statement)")

        if (
            not thr
            or not isinstance(thr, dict)
            or not all(k in thr for k in ("rationale", "source", "version"))
        ):
            errs.append("Thresholds missing rationale/source/version")

        for d in deps:
            circ_graph[sid].add(d)
        if deps and not basis:
            errs.append("Has depends_on but missing independent_validation_basis")

        per_study.append({"study_id": sid, "file": str(p), "errors": errs})

    def dfs(s, seen, stack):
        seen.add(s)
        stack.add(s)
        for nbr in circ_graph.get(s, []):
            if nbr not in seen:
                if dfs(nbr, seen, stack):
                    return True
            elif nbr in stack:
                return True
        stack.remove(s)
        return False

    cycles = False
    seen = set()
    for node in list(circ_graph.keys()):
        if node not in seen and dfs(node, seen, set()):
            cycles = True
            break

    summary = {
        "banned_phrase_hits": text_flags,
        "study_issues": per_study,
        "circular_validation_detected": cycles,
    }
    out = ROOT / "qc" / "QC_REPORT.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"[QC] Wrote {out}")
    any_err = cycles or any(s["errors"] for s in per_study) or any(text_flags)
    sys.exit(1 if any_err else 0)


if __name__ == "__main__":
    main()
