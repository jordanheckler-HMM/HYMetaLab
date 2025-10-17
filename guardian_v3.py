#!/usr/bin/env python3
"""
Guardian v3.0 — Ethical Co-Pilot Patch
Computes objectivity, sentiment, language-safety, and transparency index
across all markdown/txt docs.  Produces qc/guardian_report_v3.json and
qc/guardian_summary_v3.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
DOCS = list(ROOT.glob("**/*.md")) + list(ROOT.glob("**/*.txt"))
OUT = ROOT / "qc" / "guardian_report_v3.json"
OUTMD = ROOT / "qc" / "guardian_summary_v3.md"
OUT.parent.mkdir(parents=True, exist_ok=True)

HEDGE = {
    "suggests",
    "indicates",
    "preliminary",
    "within simulation",
    "may",
    "might",
    "could",
}
OVER = {"proves", "confirms", "undeniably", "universal law", "definitive", "conclusive"}
POS = {
    "transparent",
    "reproducible",
    "validated",
    "humble",
    "disclaimer",
    "confidence interval",
    "preregistered",
}
NEG = {"certain", "absolute", "guarantee", "prove"}


def sentiment(txt):
    t = txt.lower()
    p = sum(w in t for w in POS)
    n = sum(w in t for w in NEG)
    return (p - n) / max(1, p + n)


def objectivity(txt):
    t = txt.lower()
    h = sum(w in t for w in HEDGE)
    o = sum(w in t for w in OVER)
    return max(0, min(1, 0.6 + 0.05 * h - 0.08 * o))


def lang_safety(txt):
    o = sum(w in txt.lower() for w in OVER)
    return max(0, min(1, 1 - 0.2 * o))


def transparency():
    hits = 0
    total = 0
    for f in DOCS:
        t = f.read_text(errors="ignore")
        total += 1
        if ("doi.org" in t) or ("http" in t):
            hits += 0.4
        if "simulation" in t.lower() and (
            "hypothesis" in t.lower() or "preliminary" in t.lower()
        ):
            hits += 0.6
    return 0 if total == 0 else min(1, hits / total)


def main():
    objs = []
    sents = []
    langs = []
    for f in DOCS:
        t = f.read_text(errors="ignore")
        objs.append(objectivity(t))
        sents.append(sentiment(t))
        langs.append(lang_safety(t))
    obj = sum(objs) / max(1, len(objs))
    sen = sum(sents) / max(1, len(sents))
    lan = sum(langs) / max(1, len(langs))
    tra = transparency()
    score = round(
        100 * (0.25 * obj + 0.15 * (0.5 + 0.5 * sen) + 0.35 * tra + 0.25 * lan), 1
    )
    OUT.write_text(
        json.dumps(
            {
                "objectivity": obj,
                "sentiment": sen,
                "language_safety": lan,
                "transparency_index": tra,
                "guardian_alignment_v3": score,
            },
            indent=2,
        )
    )
    OUTMD.write_text(
        f"# Guardian v3.0 Summary\n\nScore = **{score}/100** \nObjectivity {obj:.3f} | Sentiment {sen:.3f} | LangSafety {lan:.3f} | Transparency {tra:.3f}\n"
    )
    print(f"[Guardian v3] Scanned {len(DOCS)} documents")
    print(f"[Guardian v3] Alignment Score: {score}/100")
    print(f"[Guardian v3] Report: {OUT}")
    print(f"[Guardian v3] Summary: {OUTMD}")


if __name__ == "__main__":
    main()
