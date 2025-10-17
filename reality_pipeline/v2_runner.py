import json
import os
import pathlib
from datetime import datetime

import yaml

from .ingest import load_sources
from .normalize_map import (
    build_param_pack,
    map_calibration,
    map_goals,
    map_gravity,
    map_shocks,
    map_survival,
)
from .sim_adapter import SimAdapter
from .utils import log, safe_mkdir, write_csv, write_json

THEMES = {
    "theme_3": "consciousness_calibration_index",
    "theme_6": "gravitational_consciousness_emergence",
    "theme_9": "goal_externalities_system_collapse",
}


def _timestamp_dir(base="discovery_results"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base, f"v2_{ts}")
    safe_mkdir(out)
    safe_mkdir(os.path.join(out, "data"))
    return out


def _load_cfg():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _write_reports(outdir, params, prev_dir=None):
    # Minimal auto-generated reports for themes 3,6,9
    reports = {
        "THEME_3_CONSCIOUSNESS_CALIBRATION_INDEX_REPORT_V2.md": _report_theme3(
            params, prev_dir
        ),
        "THEME_6_GRAVITATIONAL_CONSCIOUSNESS_EMERGENCE_REPORT_V2.md": _report_theme6(
            params, prev_dir
        ),
        "THEME_9_GOAL_EXTERNALITIES_SYSTEM_COLLAPSE_REPORT_V2.md": _report_theme9(
            params, prev_dir
        ),
    }
    for fname, content in reports.items():
        with open(os.path.join(outdir, fname), "w") as f:
            f.write(content)


def _maybe_prev_results():
    # Try to find any existing discovery_results folder (v1 style)
    root = pathlib.Path("discovery_results")
    if not root.exists():
        return None
    # pick the latest non-v2 folder if present
    candidates = [
        p for p in root.iterdir() if p.is_dir() and not p.name.startswith("v2_")
    ]
    return str(sorted(candidates)[-1]) if candidates else None


def _report_theme3(params, prev_dir):
    cci = params["calibration_params"]["cci"]
    prev_blob = ""
    if prev_dir:
        prev_pack = pathlib.Path(prev_dir) / "real_world_param_pack.json"
        if prev_pack.exists():
            try:
                with open(prev_pack) as f:
                    prev = json.load(f)
                pcci = prev.get("calibration_params", {}).get("cci", {})
                prev_blob = f"\n**v1 vs v2 (CCI)** — acc: {pcci.get('accuracy','?')} → {cci['accuracy']}, conf_med: {pcci.get('reported_confidence_med','?')} → {cci['reported_confidence_med']}, noise: {pcci.get('noise','?')} → {cci['noise']}\n"
            except Exception:
                pass
    return f"""# Theme 3 (V2): Consciousness Calibration Index

**Inputs (v2 mapping):**
- Median reported confidence: {cci['reported_confidence_med']:.3f}
- Accuracy (smoothed): {cci['accuracy']:.3f}
- Noise (floored): {cci['noise']:.3f}

**Notes:** v2 uses Beta-prior smoothing and a noise floor for stability.

{prev_blob}
"""


def _report_theme6(params, prev_dir):
    g = params["gravity_params"]["gravity"]
    prev_blob = ""
    if prev_dir:
        prev_pack = pathlib.Path(prev_dir) / "real_world_param_pack.json"
        if prev_pack.exists():
            try:
                with open(prev_pack) as f:
                    prev = json.load(f)
                pg = prev.get("gravity_params", {}).get("gravity", {})
                prev_blob = f"\n**v1 vs v2 (Gravity)** — softening ε: {pg.get('softening','?')} → {g['softening']}, dt: {pg.get('dt','?')} → {g['dt']}\n"
            except Exception:
                pass
    return f"""# Theme 6 (V2): Gravitational Consciousness Emergence

**Inputs (v2 mapping):**
- Softening ε: {g['softening']:.5f}
- Adaptive dt: {g['dt']:.6f}
- n bodies: {g.get('n', '50')}

**Notes:** v2 derives ε from (eccentricity + mass) and dt from orbital period / divisor to reduce energy drift.
{prev_blob}
"""


def _report_theme9(params, prev_dir):
    s = params["goals_params"]["social"]
    crm = s.get("collapse_risk_model", {})
    prev_blob = ""
    if prev_dir:
        prev_pack = pathlib.Path(prev_dir) / "real_world_param_pack.json"
        if prev_pack.exists():
            try:
                with open(prev_pack) as f:
                    prev = json.load(f)
                ps = prev.get("goals_params", {}).get("social", {})
                prev_blob = f"\n**v1 vs v2 (Collapse)** — Gini: {ps.get('gini','?')} → {s['gini']}, model: {'hard-threshold' if 'collapse_risk_model' not in ps else 'soft/logistic'} → soft/logistic, risk@Gini: {ps.get('effective_risk_at_gini','?')} → {crm.get('effective_risk_at_gini')}\n"
            except Exception:
                pass
    return f"""# Theme 9 (V2): Goal Externalities & System Collapse

**Inputs (v2 mapping):**
- Gini: {s['gini']:.3f}
- Goal diversity: {s['goal_diversity']}
- Population: {s['population']}
- Social weight: {s['social_weight']}
- Collapse risk model: logistic(center={crm.get('center')}, k={crm.get('k')}, floor={crm.get('floor')}, cap={crm.get('cap')})
- Effective risk @ Gini: {crm.get('effective_risk_at_gini'):.3f}

**Notes:** v2 replaces the hard 0.30 cutoff with a soft logistic risk curve (config-tunable).
{prev_blob}
"""


def run_selected_themes(params, outdir):
    # Domain routing to only run themes 3, 6, 9 via SimAdapter
    adapter = SimAdapter()
    avail = adapter.available()
    log.info(f"Detected modules: {avail}")

    # Theme 3: CCI / calibration
    if avail.get("calibration"):
        try:
            mod = adapter.mods.get("calibration")
            path = adapter.paths.get("calibration")
            from .utils import try_call

            ENTRYPOINTS = [("run_calibration", True), ("main", True), ("main", False)]
            try_call(
                mod, ENTRYPOINTS, params.get("calibration_params", {}), outdir, path
            )
        except Exception as e:
            log.warning(f"Theme 3 run failed: {e}")

    # Theme 6: Gravity
    if avail.get("gravity"):
        try:
            mod = adapter.mods.get("gravity")
            path = adapter.paths.get("gravity")
            from .utils import try_call

            ENTRYPOINTS = [
                ("run_gravity_experiments", True),
                ("main", True),
                ("main", False),
            ]
            try_call(mod, ENTRYPOINTS, params.get("gravity_params", {}), outdir, path)
        except Exception as e:
            log.warning(f"Theme 6 run failed: {e}")

    # Theme 9: Goals/Collapse
    if avail.get("goals") or avail.get("belief"):
        try:
            # prefer goals module; fall back to belief if that's how it's wired
            mod = adapter.mods.get("goals") or adapter.mods.get("belief")
            path = adapter.paths.get("goals") or adapter.paths.get("belief")
            from .utils import try_call

            ENTRYPOINTS = [("run_goal_sweep", True), ("main", True), ("main", False)]
            try_call(mod, ENTRYPOINTS, params.get("goals_params", {}), outdir, path)
        except Exception as e:
            log.warning(f"Theme 9 run failed: {e}")


def main():
    cfg = _load_cfg()
    outdir = _timestamp_dir(cfg.get("output_dir", "discovery_results"))

    # 1) Ingest reality data (uses refined mapping already present)
    surv_df = load_sources(cfg["survival"])
    shocks_df = load_sources(cfg["shocks"])
    goals_df = load_sources(cfg["goals"])
    calib_df = load_sources(cfg["calibration"])
    grav_df = load_sources(cfg["gravity"])

    # 2) Save raw
    write_csv(surv_df, os.path.join(outdir, "data", "survival_raw.csv"))
    write_csv(shocks_df, os.path.join(outdir, "data", "shocks_raw.csv"))
    write_csv(goals_df, os.path.join(outdir, "data", "goals_raw.csv"))
    write_csv(calib_df, os.path.join(outdir, "data", "calibration_raw.csv"))
    write_csv(grav_df, os.path.join(outdir, "data", "gravity_raw.csv"))

    # 3) Map → params with tuning from config
    params = build_param_pack(
        map_survival(surv_df),
        map_shocks(shocks_df),
        map_goals(goals_df, tuning=cfg.get("tuning", {})),
        map_calibration(calib_df, tuning=cfg.get("tuning", {})),
        map_gravity(grav_df, tuning=cfg.get("tuning", {})),
    )
    write_json(params, os.path.join(outdir, "real_world_param_pack.json"))
    log.info(f"Param pack saved → {os.path.join(outdir,'real_world_param_pack.json')}")

    # 4) Run ONLY themes 3, 6, 9
    run_selected_themes(params, outdir)

    # 5) Write v2 reports (+ v1 vs v2 deltas if possible)
    prev_dir = _maybe_prev_results()
    _write_reports(outdir, params, prev_dir)
    log.info(f"V2 theme reruns complete. Outputs at: {outdir}")


if __name__ == "__main__":
    main()
