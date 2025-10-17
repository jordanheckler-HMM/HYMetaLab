# build_paper.py
# Builds figures from sealed-lab JSON, updates the whitepaper, and exports a PDF if pandoc is available.

import json
import os
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(".")
OUT_DIR = os.path.join(ROOT, "paper")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# ---- Helper: safe read JSON from multiple candidate paths ----
def read_first_json(candidates):
    for p in candidates:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f), p
    return None, None


# ---- Locate inputs (sealed run + finalize run fallbacks) ----
observer_json, observer_path = read_first_json(
    [
        "./observer_law_summary.json",
        "./Metaphysics_Lab_Seal_summary.json",  # has nested observer_peak
        "./Metaphysics_Lab_Finalize/observer_law_summary.json",
    ]
)

evi_json, evi_path = read_first_json(
    ["./evi_summary.json", "./Metaphysics_Lab_Finalize/evi_summary.json"]
)

mus_json, mus_path = read_first_json(
    ["./mus_final_summary.json", "./Metaphysics_Lab_Finalize/mus_final_summary.json"]
)

seal_json, seal_path = read_first_json(["./Metaphysics_Lab_Seal_summary.json"])

whitepaper_md = os.path.join(OUT_DIR, "Metaphysics_Lab_Whitepaper.md")
if not os.path.exists(whitepaper_md):
    # Fallback: move/copy from root if the draft was generated there
    if os.path.exists("./Metaphysics_Lab_Whitepaper.md"):
        os.makedirs(OUT_DIR, exist_ok=True)
        shutil.copyfile("./Metaphysics_Lab_Whitepaper.md", whitepaper_md)
    else:
        print(
            "ERROR: Whitepaper markdown not found. Expected paper/Metaphysics_Lab_Whitepaper.md or ./Metaphysics_Lab_Whitepaper.md"
        )
        sys.exit(1)


# ---- Figure 1: Observer Law curve + optional CI band ----
def build_observer_figs():
    # normalize data source shape
    if observer_json is None:
        return None, None

    # When using Lab_Seal_summary.json, data is nested under 'observer_peak'
    if "observer_peak" in observer_json and isinstance(
        observer_json["observer_peak"], dict
    ):
        peak = observer_json["observer_peak"]
        densities = peak.get("agg_delta", {}).get("density", [])
        deltas = peak.get("agg_delta", {}).get("delta_CCI", [])
        rho_star = peak.get("rho_star", None)
        ci = peak.get("rho_ci", None)
    else:
        densities = observer_json.get("agg_delta", {}).get("density", [])
        deltas = observer_json.get("agg_delta", {}).get("delta_CCI", [])
        rho_star = observer_json.get("critical_density", None)
        ci = observer_json.get("rho_ci", None)

    if not densities or not deltas:
        return None, None

    # Main curve
    fig1 = os.path.join(FIG_DIR, "observer_law_curve.png")
    plt.figure(figsize=(7, 4.2))
    plt.plot(densities, deltas, marker="o")
    if rho_star is not None:
        plt.axvline(rho_star, linestyle="--")
    if ci and len(ci) == 2:
        plt.axvspan(min(ci), max(ci), alpha=0.15)
    plt.title("Observer–Coherence Law: ΔCCI vs Observation Density (ρ)")
    plt.xlabel("Observation density ρ")
    plt.ylabel("ΔCCI (collective − mean individual)")
    plt.tight_layout()
    plt.savefig(fig1, dpi=200)
    plt.close()

    # Peak zoom (±0.02 around ρ★ if available)
    fig2 = os.path.join(FIG_DIR, "observer_peak_zoom.png")
    plt.figure(figsize=(7, 4.2))
    plt.plot(densities, deltas, marker="o")
    if rho_star is not None:
        plt.axvline(rho_star, linestyle="--")
        plt.xlim(max(0, rho_star - 0.05), min(0.3, rho_star + 0.05))
    plt.title("Observer Law — Peak Region")
    plt.xlabel("Observation density ρ (zoom)")
    plt.ylabel("ΔCCI")
    plt.tight_layout()
    plt.savefig(fig2, dpi=200)
    plt.close()

    return fig1, fig2


# ---- Figure 2: Energy vs Information elasticities ----
def build_evi_bars():
    if evi_json is None:
        return None
    # We’ll display relative elasticities for E-lim vs I-lim at their baselines
    # JSON format seen: {"elasticities":{"E_lim":{"0.0005":1.0,...},"I_lim":{"0.0":1.0,...}}}
    e = evi_json.get("elasticities", {})
    e_lim = e.get("E_lim", {})
    i_lim = e.get("I_lim", {})

    # Build two bars: mean elasticity for each arm
    def mean_or_zero(d):
        vals = list(d.values())
        return sum(vals) / len(vals) if vals else 0.0

    e_mean = mean_or_zero(e_lim)
    i_mean = mean_or_zero(i_lim)

    fig = os.path.join(FIG_DIR, "evi_tradeoff_bars.png")
    plt.figure(figsize=(6, 4))
    plt.bar(["Energy-limited", "Information-limited"], [e_mean, i_mean])
    plt.title("Energy ↔ Information Elasticities (survival)")
    plt.ylabel("Elasticity (unitless)")
    plt.tight_layout()
    plt.savefig(fig, dpi=200)
    plt.close()
    return fig


# ---- Figure 3: MUS constants card (simple text visual) ----
def build_mus_card():
    if mus_json is None:
        return None
    coh = mus_json.get("coh_unit", None)
    cal = mus_json.get("cal_scale", None)
    em = mus_json.get("em_scale", None)
    eps = mus_json.get("eps_unit", None)
    text = f"MUS Constants\n\ncoh_unit: {coh}\ncal_scale: {cal}\nem_scale: {em}\neps_unit: {eps}\n"
    fig = os.path.join(FIG_DIR, "mus_constants.png")
    plt.figure(figsize=(6, 4))
    plt.axis("off")
    plt.text(0.02, 0.98, text, va="top")
    plt.tight_layout()
    plt.savefig(fig, dpi=200)
    plt.close()
    return fig


# ---- Build everything ----
obs_curve, obs_zoom = build_observer_figs()
evi_bars = build_evi_bars()
mus_card = build_mus_card()

# ---- Inject figures into the Markdown (append at the right sections if tags exist; else append at end) ----
with open(whitepaper_md, encoding="utf-8") as f:
    md = f.read()


def insert_after_heading(md_text, heading, img_markdown):
    # Find heading line like "5. Results" or "6. Discussion & Interpretation"
    lines = md_text.splitlines()
    out = []
    inserted = False
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if not inserted and lines[i].strip().lower().startswith(heading.lower()):
            # insert right after a blank line following the heading if present
            out.append("")
            out.append(img_markdown)
            out.append("")
            inserted = True
        i += 1
    if not inserted:
        out.append("\n" + img_markdown + "\n")
    return "\n".join(out)


def img_block(title, path_rel):
    return f"**Figure — {title}.**\n\n![]({os.path.relpath(path_rel, OUT_DIR).replace(os.sep,'/')})"


inserts = []

if obs_curve:
    inserts.append(
        ("5. Results", img_block("Observer–Coherence Law (ΔCCI vs ρ)", obs_curve))
    )
if obs_zoom:
    inserts.append(("5. Results", img_block("Observer Law — Peak Region", obs_zoom)))
if evi_bars:
    inserts.append(
        ("5. Results", img_block("Energy ↔ Information Elasticities", evi_bars))
    )
if mus_card:
    inserts.append(("3. Theoretical Framework", img_block("MUS Constants", mus_card)))

updated = md
for heading, block in inserts:
    updated = insert_after_heading(updated, heading, block)

with open(whitepaper_md, "w", encoding="utf-8") as f:
    f.write(updated)

print(f"Updated whitepaper with figures at: {whitepaper_md}")

# ---- Try to export to PDF via pandoc if available ----
PDF_OUT = os.path.join(OUT_DIR, "Metaphysics_Lab_Whitepaper.pdf")


def has_pandoc():
    try:
        subprocess.run(
            ["pandoc", "-v"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


if has_pandoc():
    try:
        subprocess.run(
            [
                "pandoc",
                whitepaper_md,
                "-o",
                PDF_OUT,
                "--from",
                "markdown",
                "--pdf-engine=xelatex",
                "--metadata",
                "title=Metaphysics Lab Whitepaper",
            ],
            check=True,
        )
        print(f"PDF exported: {PDF_OUT}")
    except Exception as e:
        print(
            "Pandoc found but PDF export failed. You can still open the Markdown or install a LaTeX engine."
        )
        print(e)
else:
    print(
        "Pandoc not found; PDF not generated. Install pandoc + LaTeX (or export from VS Code Markdown PDF)."
    )
