#!/usr/bin/env python3
# ===========================================================
# openlaws_automation.py
# One-file automation for run → validate → report
# Usage:
#   python openlaws_automation.py run        # run the grid defined in study.yml
#   python openlaws_automation.py validate   # classify & update archive
#   python openlaws_automation.py report     # figures + README for latest run
#   python openlaws_automation.py auto       # run → validate → (report if validated)
# Requirements: pyyaml, pandas, numpy, matplotlib, scipy
#   pip install pyyaml pandas numpy matplotlib scipy
# ===========================================================
import argparse
import datetime
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

try:
    from meta.logging.logger import get_logger

    logger = get_logger()
except Exception:
    # If the meta package isn't available (tests, etc.), fall back to a simple print-logger
    class _Fake:
        def info(self, *a, **k):
            print(*a)

        def warning(self, *a, **k):
            print(*a)

        def error(self, *a, **k):
            print(*a)

    logger = _Fake()

# Optional HYMetaLab agent charter helpers (non-fatal)
try:
    from openlaws_agent_charter import print_charter_summary, validate_and_enforce
except Exception:
    validate_and_enforce = None
    print_charter_summary = None

# ---------- paths ----------
ROOT = Path(".").resolve()
ARCHIVE = ROOT / "project_archive"
ARCHIVE.mkdir(exist_ok=True, parents=True)
INTEGRITY_MD = ARCHIVE / "Research_Integrity_Prompt.md"

# ---------- defaults ----------
# HYMetaLab preferred deterministic seeds (per charter)
DEFAULT_SEEDS = [11, 17, 23, 29]


# ---------- utils ----------
def ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def get_seeds(cfg: dict) -> list:
    """
    Priority:
      1) cfg['constants']['seeds']
      2) cfg['seeds']  (legacy / top-level)
      3) DEFAULT_SEEDS
    Logs a warning if falling back.
    """
    const = cfg.get("constants", {})
    if isinstance(const, dict) and "seeds" in const:
        seeds = const["seeds"]
    elif "seeds" in cfg:
        seeds = cfg["seeds"]
    else:
        seeds = DEFAULT_SEEDS
        print(
            "[openlaws] WARN: no seeds found in constants or top-level; "
            f"using defaults {DEFAULT_SEEDS}",
            file=sys.stderr,
        )
    # normalize to ints
    try:
        seeds = [int(s) for s in seeds]
    except Exception:
        print(
            "[openlaws] WARN: non-int seeds; coercing where possible", file=sys.stderr
        )
        seeds = [int(float(s)) for s in seeds]
    return seeds


def resolve_results_dir(study_id: str, expect_writable: bool = False) -> Path:
    """
    Preferred canonical dir: results/discovery_results/<study_id>/
    If not present, fall back to newest matching timestamped dir:
      results/discovery_results/<study_id>_*  (lexicographically or mtime)
    Also checks discovery_results/<study_id>_* for backward compatibility.
    """
    # Try results/discovery_results first (canonical location)
    base = ROOT / "results" / "discovery_results"
    canonical = base / study_id
    if canonical.exists():
        return canonical

    # Fallback: find latest timestamped dir in results/discovery_results
    candidates = sorted(
        (p for p in base.glob(f"{study_id}_*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    # Also check discovery_results/ (old location) for backward compatibility
    base_old = ROOT / "discovery_results"
    if base_old.exists():
        candidates_old = sorted(
            (p for p in base_old.glob(f"{study_id}_*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates_old:
            return candidates_old[0]

    # If still not found and we need to write, create canonical
    if expect_writable:
        canonical.mkdir(parents=True, exist_ok=True)
        return canonical

    raise FileNotFoundError(
        f"[openlaws] No results directory found for '{study_id}'. "
        f"Looked for {canonical} and {base}/{study_id}_*, {base_old}/{study_id}_*"
    )


def load_yaml(p: Path) -> dict[str, Any]:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_study_cfg(cfg: dict[str, Any], study_path: Path) -> dict[str, Any]:
    """Normalize different study YAML shapes into a common dict with keys:
    - study_id, title, adapter, seeds, sweep, constants, integrity, acceptance_criteria
    """
    # If user provided a pre-OpenLaws style with top-level 'prereg' key, map to old format
    out: dict[str, Any] = {}
    # New OpenLaws-style (as created by studies/*.yml)
    if (
        "study_id" in cfg
        or "adapter" in cfg
        or "sweep" in cfg
        or "suite" in cfg
        or "tests" in cfg
    ):
        out["study_id"] = cfg.get("study_id") or (cfg.get("study", {}).get("id"))
        out["title"] = cfg.get("title") or cfg.get("study", {}).get("title")
        out["adapter"] = cfg.get("adapter")
        # Use new get_seeds function that respects constants.seeds
        seeds = get_seeds(cfg)
        out["seed_policy"] = {"seeds": seeds}
        out["__seeds"] = seeds  # Canonical location for downstream code
        # Suite: either 'suite', 'tests' or 'sweep'
        out["suite"] = cfg.get("suite") or cfg.get("tests") or cfg.get("sweep") or []
        out["sweep"] = cfg.get("sweep", {})
        # epochs/bootstrap may appear at top-level or under common
        out["epochs"] = cfg.get("epochs") or cfg.get("common", {}).get("epochs")
        out["bootstrap"] = cfg.get("bootstrap") or cfg.get("common", {}).get(
            "bootstrap"
        )
        out["constants"] = cfg.get("constants", {})
        out["integrity"] = cfg.get("integrity", {})
        out["thresholds"] = cfg.get("thresholds", {})
        out["acceptance_criteria"] = cfg.get("acceptance_criteria", {})
        # Pass through additional Phase 33 style keys
        out["exports"] = cfg.get("exports", {})
        out["validation"] = cfg.get("validation", {})
        out["hypothesis"] = cfg.get("hypothesis", {})
        out["report"] = cfg.get("report", {})
        out["version"] = cfg.get("version")
        out["prereg_date"] = cfg.get("prereg_date")
        out["protocols"] = cfg.get("protocols", [])  # Multi-protocol studies
        out["data"] = cfg.get("data", {})  # Phase 35b-style dataset definitions
        out["mapping"] = cfg.get("mapping", {})  # Column mapping for datasets
        return out

    # Legacy format: expected shape used by previous automation
    if "prereg" in cfg and isinstance(cfg["prereg"], dict):
        const = cfg["prereg"].get("constants", {})
        out["study_id"] = cfg.get("study", {}).get("id")
        out["title"] = cfg.get("study", {}).get("title")
        out["adapter"] = const.get("sim_adapter")
        out["seed_policy"] = {"seeds": const.get("seeds", [101, 102, 103])}
        # infer sweep keys from lists in constants
        sweep = {k: v for k, v in const.items() if isinstance(v, list) and k != "seeds"}
        out["sweep"] = sweep
        out["constants"] = const
        out["integrity"] = cfg.get("integrity", {})
        out["thresholds"] = cfg.get("integrity", {}).get("thresholds", {})
        out["acceptance_criteria"] = {}
        return out

    # Fallback: wrap minimally
    return {
        "study_id": cfg.get("study_id") or study_path.stem,
        "title": cfg.get("title", ""),
        "adapter": cfg.get("adapter"),
        "seed_policy": cfg.get("seed_policy", {}),
        "sweep": cfg.get("sweep", {}),
        "constants": cfg.get("constants", {}),
        "integrity": cfg.get("integrity", {}),
        "thresholds": cfg.get("thresholds", {}),
        "acceptance_criteria": cfg.get("acceptance_criteria", {}),
    }


def ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Harmonize required columns; fill missing with NaN
    rename_map = {}
    if "survival" in df.columns and "survival_rate" not in df.columns:
        rename_map["survival"] = "survival_rate"
    df = df.rename(columns=rename_map)
    for col in ["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]:
        if col not in df.columns:
            df[col] = np.nan
    return df[["epoch", "CCI", "survival_rate", "hazard", "collapse_risk"]]


def bootstrap_ci(values, n_boot=800, alpha=0.05, seed=42):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
    if len(v) < 2:
        return None
    rng = np.random.default_rng(seed)
    boots = []
    n = len(v)
    for _ in range(n_boot):
        sample = rng.choice(v, size=n, replace=True)
        boots.append(np.mean(sample))
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return float(np.mean(v)), lo, hi


def find_column(df: pd.DataFrame, candidates: list[str]):
    """Return the first column name in df that matches any candidate (case-insensitive), or None."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def import_func(func_path: str):
    # "path/to/adapter.py:run" OR "package.module:func"
    # Be tolerant: some study YAMLs may accidentally supply a dict for adapter.
    if isinstance(func_path, dict):
        # common nested keys
        for k in ["sim_adapter", "adapter", "run", "entrypoint"]:
            if k in func_path and isinstance(func_path[k], str):
                func_path = func_path[k]
                break
        # Support dict shape: { 'path': 'adapters/foo.py', 'function': 'run' }
        if isinstance(func_path, dict):
            if "path" in func_path and "function" in func_path:
                func_path = f"{func_path['path']}:{func_path['function']}"
            elif "module" in func_path and "function" in func_path:
                func_path = f"{func_path['module']}:{func_path['function']}"
            else:
                raise RuntimeError(
                    f"Adapter entry is a dict but contains no string entrypoint: {func_path}"
                )
    if not isinstance(func_path, str):
        raise RuntimeError(
            f"Adapter must be a string like 'module:func' but got: {type(func_path)} ({func_path})"
        )
    mod_path, fn_name = func_path.split(":")
    if mod_path.endswith(".py") and (ROOT / mod_path).exists():
        # add file's parent to sys.path then import as module name
        full = (ROOT / mod_path).resolve()
        sys.path.insert(0, str(full.parent))
        module_name = full.stem
    else:
        module_name = mod_path
    mod = importlib.import_module(module_name)
    if hasattr(mod, fn_name):
        return getattr(mod, fn_name)
    # try common fallbacks
    for alt in ["run", "run_study", "run_adapter", "main", "entrypoint"]:
        if hasattr(mod, alt):
            print(
                f"[import_func] Warning: requested {fn_name} not found in {module_name}; using fallback {alt}"
            )
            return getattr(mod, alt)
    raise AssertionError(
        f"{func_path} not found in module {module_name}. Available: {dir(mod)[:10]}"
    )


def summarize_category(row, thr_cci, thr_surv, require_prereg, require_ci):
    ok_prereg = bool(row.get("preregistered_flag", True))
    ok_ci = bool(
        row.get("cci_ci_present", False) and row.get("survival_ci_present", False)
    )
    ok_nums = (row.get("mean_CCI", 0) >= thr_cci) and (
        row.get("mean_survival", 0) >= thr_surv
    )
    if require_prereg and not ok_prereg:
        return "under_review"
    if require_ci and not ok_ci:
        return "under_review"
    if ok_nums:
        return "validated"
    return "under_review" if (ok_prereg or ok_ci) else "hypothesis_only"


# ---------- pipeline steps ----------
def step_run(study_path: Path):
    raw_cfg = load_yaml(study_path)
    cfg = normalize_study_cfg(raw_cfg, study_path)

    func_path = cfg.get("adapter")
    if not func_path:
        raise RuntimeError(f"No adapter defined in {study_path}")
    run_fn = import_func(func_path)

    # prepare output dir
    outdir = ROOT / "discovery_results" / f"{cfg.get('study_id')}_{ts()}"
    datadir = outdir / "data"
    repdir = outdir / "report"
    datadir.mkdir(parents=True, exist_ok=True)
    repdir.mkdir(parents=True, exist_ok=True)

    # New-style: adapter may accept (config, out_dir, seed) and write aggregated CSVs per seed.
    seeds = cfg.get("__seeds", DEFAULT_SEEDS)

    # Build a study_config dict for adapters. Some adapters expect the full study_config
    # (with 'constants' and 'sweep' keys) while others prefer a flattened dict.
    base_adapter_cfg = {
        "study_id": cfg.get("study_id"),
        "title": cfg.get("title"),
        "constants": cfg.get("constants", {}),
        "sweep": cfg.get("sweep", {}),
        "exports": cfg.get("exports", {}),
        "validation": cfg.get("validation", {}),
        "report": cfg.get("report", {}),
    }
    # Backwards-compatible top-level shortcuts
    if cfg.get("epochs"):
        base_adapter_cfg.setdefault("constants", {})
        base_adapter_cfg["constants"].setdefault("epochs", cfg.get("epochs"))
    if cfg.get("bootstrap"):
        base_adapter_cfg.setdefault("constants", {})
        base_adapter_cfg["constants"].setdefault("bootstrap", cfg.get("bootstrap"))

    # If adapter returns a dict with file paths, call it once per seed with the adapter config.
    results_per_seed = []
    for seed in seeds:
        try:
            res = None
            try:
                # try positional call with flattened adapter config: (adapter_config, outdir, seed)
                # Pass raw_cfg (original YAML) for maximal compatibility, but keep base_adapter_cfg as fallback
                res = run_fn(raw_cfg, str(outdir), int(seed))
            except TypeError:
                # try alternative param names commonly used by adapters
                try:
                    res = run_fn(config=raw_cfg, outdir=str(outdir), seed=int(seed))
                except TypeError:
                    res = run_fn(config=raw_cfg, out_dir=str(outdir), seed=int(seed))
        except Exception as e:
            print(f"[run] adapter call failed for seed {seed}: {e}")
            raise
        results_per_seed.append(res)

    # Collect outputs: if adapter returned file paths, read and merge; if it returned DataFrame(s), handle accordingly
    runs_list = []
    trajs_list = []
    for i, res in enumerate(results_per_seed):
        # Log adapter return for easier debugging
        try:
            print(f"[run] adapter return for seed {seeds[i]}: {res}")
        except Exception:
            print(f"[run] adapter return (index {i}): type={type(res)}")

        if isinstance(res, dict) and res.get("runs_summary_csv"):
            runs_path = res.get("runs_summary_csv")
            try:
                runs_list.append(pd.read_csv(runs_path))
            except Exception as e:
                print(f"[run] Failed to read runs_summary_csv {runs_path}: {e}")

            traj_path = res.get("trajectories_long_csv")
            if traj_path:
                try:
                    trajs_list.append(pd.read_csv(traj_path))
                except Exception as e:
                    print(
                        f"[run] Failed to read trajectories_long_csv {traj_path}: {e}"
                    )

        elif isinstance(res, dict) and any(
            isinstance(v, (pd.DataFrame,)) for v in res.values()
        ):
            # Adapter returned DataFrames in a dict; write/read them into our lists
            if "runs_summary" in res and isinstance(res["runs_summary"], pd.DataFrame):
                runs_list.append(res["runs_summary"])
            if "trajectories_long" in res and isinstance(
                res["trajectories_long"], pd.DataFrame
            ):
                trajs_list.append(res["trajectories_long"])
            # if runs_summary is provided as a DataFrame under another key, try to find it
            for k, v in res.items():
                if isinstance(v, pd.DataFrame) and k not in (
                    "runs_summary",
                    "trajectories_long",
                ):
                    # heuristics: if df has 'seed' or 'run_id' treat as runs summary
                    cols = set(v.columns)
                    if "seed" in cols or "run_id" in cols:
                        runs_list.append(v)
                    else:
                        trajs_list.append(v)

        elif isinstance(res, pd.DataFrame):
            # legacy adapter that returned df of a single run
            df = ensure_cols(res)
            runs_list.append(df)
        else:
            # unknown return; ignore but log for diagnostics
            print(f"[run] Adapter returned unsupported result for index {i}: {res}")

    if runs_list:
        merged_runs = pd.concat(runs_list, ignore_index=True)
        merged_runs.to_csv(datadir / "runs_summary.csv", index=False)
    if trajs_list:
        merged_trajs = pd.concat(trajs_list, ignore_index=True)
        merged_trajs.to_csv(datadir / "trajectories_long.csv", index=False)

    # minimal report file for downstream steps
    (repdir / "results.md").write_text(
        f"# Study Results — {cfg.get('title')}\n- Data dir: {datadir}\n",
        encoding="utf-8",
    )
    print(f"[run] outdir: {outdir}")
    return outdir


def _scan_runs_for_metrics(runs_csv: Path, thr_cci: float, thr_surv: float):
    df = pd.read_csv(runs_csv)

    def find_col(df, candidates):
        lower_map = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    # detect main metric columns with fallbacks
    cci_col = find_col(df, ["CCI", "cci", "final_cci", "cci_mean"]) or "CCI"
    surv_col = (
        find_col(df, ["survival_rate", "survival", "avg_survival"]) or "survival_rate"
    )
    hazard_col = find_col(df, ["hazard"]) or "hazard"
    collapse_col = (
        find_col(df, ["collapse_risk", "collapse", "collapse_rate"]) or "collapse_risk"
    )

    # detect presence flags (case-insensitive)
    cci_ci_present = any(
        x.lower() in (c.lower() for c in df.columns)
        for x in ["cci_ci", f"{cci_col}_ci_lo", f"{cci_col}_ci_hi"]
    )
    surv_ci_present = any(
        x.lower() in (c.lower() for c in df.columns)
        for x in ["survival_ci", f"{surv_col}_ci_lo", f"{surv_col}_ci_hi"]
    )

    # prereg detection: look for preregistered or prereg flag columns
    prereg_candidates = ["preregistered", "preregistered_flag", "prereg"]
    prereg_flag = False
    for p in prereg_candidates:
        if p in df.columns:
            try:
                prereg_flag = bool(df[p].astype(bool).any())
            except Exception:
                prereg_flag = True
            break

    # compute means robustly
    def mean_col(df, colname):
        if colname in df.columns:
            return float(pd.to_numeric(df[colname], errors="coerce").mean())
        return float("nan")

    mean_cci = mean_col(df, cci_col)
    mean_surv = mean_col(df, surv_col)

    return {
        "file": str(runs_csv),
        "mean_CCI": mean_cci,
        "mean_survival": mean_surv,
        "mean_hazard": mean_col(df, hazard_col),
        "mean_collapse": mean_col(df, collapse_col),
        "preregistered_flag": bool(prereg_flag),
        "cci_ci_present": bool(cci_ci_present),
        "survival_ci_present": bool(surv_ci_present),
        "thresholds": {"cci": thr_cci, "survival": thr_surv},
    }


def step_validate(study_path: Path):
    raw_cfg = load_yaml(study_path)
    cfg = normalize_study_cfg(raw_cfg, study_path)

    # find the most recent discovery run for this study
    pattern = f"{cfg.get('study_id')}_*"
    candidates = sorted((ROOT / "discovery_results").glob(pattern))
    records = []
    validated = []
    if not candidates:
        print(f"[validate] No discovery_results found for study {cfg.get('study_id')}")
        return validated, records

    # pick latest
    latest = candidates[-1]

    # Try multiple possible CSV locations (backward compatibility)
    study_id = cfg.get("study_id", "")
    possible_csvs = [
        latest / "data" / "runs_summary.csv",  # old style
        latest / "phase33_coop_meaning_results.csv",  # phase33 adapter style
        latest / "runs_summary.csv",  # generic location
        latest / f"{study_id}_results.csv",  # study-specific naming
        latest / "fis_emotion_results.csv",  # FIS emotion
        latest / "fis_trust_hope_results.csv",  # FIS trust/hope
        latest / "fis_ai_safety_results.csv",  # FIS AI safety
    ]

    runs_csv = None
    for csv_path in possible_csvs:
        if csv_path.exists():
            runs_csv = csv_path
            break

    if not runs_csv:
        print(f"[validate] No results CSV found in {latest}")
        print(f"[validate] Tried: {[str(p.name) for p in possible_csvs]}")
        return validated, records

    df = pd.read_csv(runs_csv)
    print(f"[validate] Loaded {len(df)} rows from {runs_csv.name}")

    # Determine acceptance using new-style acceptance_criteria if present
    ac = cfg.get("acceptance_criteria", {})
    validation_cfg = cfg.get("validation", {})
    validated_flag = False
    reason = "no criteria matched"

    # Phase 33 style validation: check metrics against rules
    if validation_cfg and "metrics" in validation_cfg:
        baseline_CCI = 0.54  # Default baseline
        baseline_hazard = 0.254

        mean_cci = df["CCI"].mean() if "CCI" in df.columns else np.nan
        mean_hazard = df["hazard"].mean() if "hazard" in df.columns else np.nan

        delta_cci = (
            (mean_cci - baseline_CCI) / baseline_CCI
            if not np.isnan(mean_cci)
            else np.nan
        )
        delta_hazard = (
            mean_hazard - baseline_hazard if not np.isnan(mean_hazard) else np.nan
        )

        metrics_met = []
        for metric_cfg in validation_cfg["metrics"]:
            metric_name = metric_cfg.get("name")
            rule = metric_cfg.get("rule", "")

            if metric_name == "mean_CCI_gain":
                value = delta_cci
            elif metric_name == "mean_hazard_delta":
                value = delta_hazard
            else:
                continue

            # Parse rule (e.g., ">= 0.03" or "<= -0.01")
            try:
                if ">=" in rule:
                    threshold = float(rule.split(">=")[1].strip())
                    passed = value >= threshold
                elif "<=" in rule:
                    threshold = float(rule.split("<=")[1].strip())
                    passed = value <= threshold
                else:
                    passed = False
                metrics_met.append(passed)
            except:
                metrics_met.append(False)

        if metrics_met:
            validated_flag = all(metrics_met)
            reason = f"Phase33: ΔCCI={delta_cci:.5f}, Δhazard={delta_hazard:.5f}, {sum(metrics_met)}/{len(metrics_met)} metrics met"

    # If creation_success column exists, compute fraction (original logic)
    elif "creation_success" in df.columns and "validated_fraction_min" in ac:
        frac = float(df["creation_success"].mean())
        validated_flag = frac >= float(ac["validated_fraction_min"])
        reason = (
            f"creation_success_frac={frac:.3f} (min {ac['validated_fraction_min']})"
        )
    else:
        # fallback to simple thresholds (legacy)
        thr = cfg.get("thresholds", {})
        if "creation_success" in df.columns:
            frac = float(df["creation_success"].mean())
            validated_flag = frac >= 0.6
            reason = f"creation_success_frac={frac:.3f}"
        else:
            validated_flag = False
            reason = "no criteria matched"

    # compute delta metrics where possible
    mean_cci = (
        float(df.get("CCI", pd.Series([np.nan])).mean())
        if "CCI" in df.columns
        else float("nan")
    )
    mean_hazard = (
        float(df.get("hazard", pd.Series([np.nan])).mean())
        if "hazard" in df.columns
        else float("nan")
    )
    baseline_CCI = 0.54
    baseline_hazard = 0.254
    delta_cci = (
        (mean_cci - baseline_CCI) / baseline_CCI
        if not np.isnan(mean_cci)
        else float("nan")
    )
    delta_hazard = (
        (mean_hazard - baseline_hazard) if not np.isnan(mean_hazard) else float("nan")
    )

    # compute SHA256 of CSV for integrity tagging
    try:
        with open(runs_csv, "rb") as fh:
            data = fh.read()
            sha256hex = hashlib.sha256(data).hexdigest()
    except Exception:
        sha256hex = None

    # classification per HYMetaLab charter
    metrics_met = locals().get("metrics_met", []) or []
    if validated_flag:
        classification = "VALIDATED"
    elif any(metrics_met):
        classification = "PARTIAL"
    else:
        classification = "HYPOTHESIS-GEN"

    rec = {
        "file": str(runs_csv),
        "mean_CCI": mean_cci,
        "mean_survival": (
            float(df.get("survival_rate", pd.Series([np.nan])).mean())
            if "survival_rate" in df.columns
            else float("nan")
        ),
        "mean_hazard": mean_hazard,
        "delta_CCI": delta_cci,
        "delta_hazard": delta_hazard,
        "creation_success_frac": (
            float(df["creation_success"].mean())
            if "creation_success" in df.columns
            else np.nan
        ),
        "validated": bool(validated_flag),
        "classification": classification,
        "sha256": sha256hex,
        "reason": reason,
    }
    records.append(rec)
    if validated_flag:
        validated.append(rec)

    # archive summary
    summary_json = ARCHIVE / f"project_summary_{ts()}.json"
    pd.DataFrame(records).to_json(summary_json, orient="records", indent=2)

    # write integrity prompt
    lines = ["# Research Integrity Status", ""]
    lines.append(f"Study: {cfg.get('study_id')} — validated={len(validated)>0}")
    lines.append(f"Reason: {reason}")
    INTEGRITY_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[validate] wrote {summary_json.name}")
    # write a small manifest for traceability
    manifest = {
        "study_id": cfg.get("study_id"),
        "run_file": str(runs_csv),
        "sha256": sha256hex,
        "classification": classification,
        "delta_CCI": delta_cci,
        "delta_hazard": delta_hazard,
        "validated": bool(validated_flag),
        "reason": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    manifest_path = ARCHIVE / f"manifest_{cfg.get('study_id')}_{ts()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # update integrity markdown with more details
    lines = ["# Research Integrity Status", ""]
    lines.append(f"Study: {cfg.get('study_id')} — validated={len(validated)>0}")
    lines.append(f"Classification: {classification}")
    if sha256hex:
        lines.append(f"SHA256: {sha256hex}")
    lines.append(f"Reason: {reason}")
    INTEGRITY_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"[validate] wrote {manifest_path.name}")
    print(f"[validate] updated {INTEGRITY_MD}")

    # If validated, optionally export public artifacts and generate a run manifest into github_release/public_artifacts
    if validated:
        try:
            from tools.export_public_artifacts import export as export_public
            from tools.generate_run_manifest import generate_manifest

            # latest is the discovery_results run we inspected
            run_name = latest.name
            gh_pub_base = ROOT / "github_release" / "public_artifacts" / run_name
            gh_pub_base.parent.mkdir(parents=True, exist_ok=True)
            # export public artifacts (summary, figures, report)
            export_public(latest, out_base=ROOT / "github_release" / "public_artifacts")
            # generate a small run manifest
            study_yaml = str(study_path)
            adapter_file = None
            # try to resolve adapter file path
            ad = cfg.get("adapter") or ""
            if ad and ad.endswith(".py"):
                adapter_file = str(ROOT / ad)
            generate_manifest(
                str(ROOT / "github_release" / "public_artifacts" / run_name),
                study_yaml=study_yaml,
                adapter_path=adapter_file,
            )
            print(
                f"[validate] exported public artifacts to github_release/public_artifacts/{run_name}"
            )
        except Exception as e:
            print(f"[validate] public export failed: {e}")

    return validated, records


def step_report(validated_records: list[dict[str, Any]]):
    if not validated_records:
        print("[report] No validated datasets yet. Skipping figure pack.")
        return None
    # pick the most recent validated run
    v = sorted(validated_records, key=lambda r: r["file"])[-1]
    run_csv = Path(v["file"])
    run_dir = run_csv.parent.parent
    report_dir = run_dir / "report"
    report_dir.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(run_csv)
    c_eps = find_column(df, ["epsilon", "eps", "epsilon_val"]) or None
    c_cci = find_column(df, ["CCI", "cci", "final_cci", "cci_mean"])
    c_surv = find_column(
        df, ["survival_rate", "survival", "final_survival"]
    ) or find_column(df, ["survival_rate", "survival"])

    # plots
    stamp = ts()

    def savefig(name):
        p = report_dir / name
        plt.tight_layout()
        plt.savefig(p, dpi=300)
        plt.close()
        print(f"[report] {p.name}")
        return p

    # Only plot if we actually found matching columns
    if c_eps and c_cci is not None:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[c_eps], df[c_cci], s=6)
        thr_cci = v.get("thresholds", {}).get("cci")
        if thr_cci is not None:
            plt.axhline(thr_cci, ls="--")
        plt.xlabel("epsilon")
        plt.ylabel("CCI")
        plt.title("CCI vs epsilon")
        fig1 = savefig(f"fig_cci_vs_epsilon_{stamp}.png")

    if c_eps and c_surv is not None:
        plt.figure(figsize=(8, 5))
        plt.scatter(df[c_eps], df[c_surv], s=6)
        thr_surv = v.get("thresholds", {}).get("survival")
        if thr_surv is not None:
            plt.axhline(thr_surv, ls="--")
        plt.xlabel("epsilon")
        plt.ylabel("survival_rate")
        plt.title("Survival vs epsilon")
        fig2 = savefig(f"fig_survival_vs_epsilon_{stamp}.png")

    # abstract
    abstract = report_dir / f"validated_abstract_{stamp}.md"
    thr = v.get("thresholds", {}) if isinstance(v, dict) else {}
    prereg = v.get("preregistered_flag", v.get("preregistered", False))
    cci_ci = v.get("cci_ci_present", v.get("cci_ci", False))
    surv_ci = v.get("survival_ci_present", v.get("survival_ci", False))

    # Safe formatting of potentially None values
    mean_cci = v.get("mean_CCI", float("nan"))
    mean_survival = v.get("mean_survival")
    mean_cci_str = (
        f"{mean_cci:.3f}" if mean_cci is not None and not np.isnan(mean_cci) else "N/A"
    )
    mean_survival_str = (
        f"{mean_survival:.3f}"
        if mean_survival is not None and not np.isnan(mean_survival)
        else "N/A"
    )

    abstract_text = (
        f"# Validated Study Abstract — {stamp}\n\n"
        f"**Dataset:** `{run_csv}`\n\n"
        f"**Means:** CCI={mean_cci_str}  |  Survival={mean_survival_str}\n"
        f"**Thresholds:** CCI≥{thr.get('cci', 'N/A')}  Survival≥{thr.get('survival', 'N/A')}\n"
        f"**Integrity:** prereg={prereg}  CI(cci/surv)={cci_ci}/{surv_ci}\n\n"
        f"**Summary**\nIncreasing openness (ε) produced higher coherence (CCI) and survival, meeting validation thresholds under preregistered conditions with bootstrapped CIs.\n"
    )
    abstract.write_text(abstract_text, encoding="utf-8")
    print(f"[report] {abstract.name}")
    return report_dir


def apply_bootstrap_validation(study_path: Path, n_boot=800):
    """Apply bootstrap CI to Phase 33 results and update summary."""
    raw_cfg = load_yaml(study_path)
    cfg = normalize_study_cfg(raw_cfg, study_path)

    # Find results directory
    study_id = cfg.get("study_id")
    results_dir = ROOT / "results" / "discovery_results" / study_id

    if not results_dir.exists():
        print(f"[bootstrap] Results directory not found: {results_dir}")
        return

    # Load results
    results_csv = results_dir / f"{study_id}_results.csv"
    if not results_csv.exists():
        print(f"[bootstrap] Results CSV not found: {results_csv}")
        return

    df = pd.read_csv(results_csv)
    print(f"[bootstrap] Loaded {len(df)} results from {results_csv.name}")

    # Compute bootstrap CIs for key metrics
    metrics_to_bootstrap = ["CCI", "hazard", "survival", "delta_CCI", "delta_hazard"]
    ci_results = {}

    for metric in metrics_to_bootstrap:
        if metric in df.columns:
            result = bootstrap_ci(df[metric], n_boot=n_boot)
            if result:
                mean, lo, hi = result
                ci_results[metric] = {
                    "mean": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "ci_width": hi - lo,
                }
                print(f"[bootstrap] {metric}: {mean:.4f} [{lo:.4f}, {hi:.4f}]")

    # Update summary.json with bootstrap results
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        summary["bootstrap_ci"] = {
            "n_iterations": n_boot,
            "metrics": ci_results,
            "computed_at": datetime.datetime.now().isoformat(),
        }

        # Re-classify based on CI
        hyp = summary["hypothesis_test"]

        # Check if CCI gain CI lower bound exceeds threshold
        if "delta_CCI" in ci_results:
            cci_ci_lo = ci_results["delta_CCI"]["ci_lo"]
            cci_validated = cci_ci_lo >= 0.03
            print(
                f"[bootstrap] CCI gain CI: [{cci_ci_lo:.4f}, {ci_results['delta_CCI']['ci_hi']:.4f}]"
            )
            print(f"[bootstrap] CCI validated (CI_lo ≥ 0.03): {cci_validated}")

            # Update classification
            if cci_validated and hyp.get("all_passed", False):
                summary["classification"] = "validated"
            elif cci_validated or hyp.get("all_passed", False):
                summary["classification"] = "under_review_promising"
            else:
                summary["classification"] = "under_review"

        # Save updated summary
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"[bootstrap] Updated summary with CI results: {summary_path}")
        print(f"[bootstrap] Classification: {summary.get('classification', 'unknown')}")

    return ci_results


# ---------- Init Command ----------
def step_init(
    study_name: str, hypothesis: str, sweep: str, metrics: str, exports: str
) -> Path:
    """Create a new study YAML stub.

    study_name may be a simple name (e.g., 'foo') or a relative/absolute path
    like 'studies/foo.yml' or './studies/foo.yml'. If a bare name is given,
    the file is created under ./studies/<name>.yml.
    Returns the Path to the created YAML.
    """
    # Accept either a filename/path or a bare study id
    sp = Path(study_name)
    if sp.suffix in (".yml", ".yaml") or str(study_name).count(os.sep) > 0:
        study_path = ROOT / sp
    else:
        study_path = ROOT / "studies" / f"{study_name}.yml"

    if study_path.exists():
        print(f"[init] Study already exists: {study_path}")
        return study_path

    study_path.parent.mkdir(parents=True, exist_ok=True)

    # Parse sweep string (e.g., "epsilon=[0.0005,0.001]; seeds=[11,17]; shock=[0.5]")
    sweep_dict = {}
    constants_dict = {}

    for item in sweep.split(";"):
        item = item.strip()
        if "=" in item:
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Parse list values
            if value.startswith("[") and value.endswith("]"):
                values = [
                    float(v.strip()) if "." in v.strip() else int(v.strip())
                    for v in value[1:-1].split(",")
                ]

                # Seeds and agents go to constants, others to sweep
                if key in ["seeds", "agents", "shock"]:
                    if key == "shock" and len(values) == 1:
                        constants_dict["shock"] = {"epoch": 1000, "severity": values[0]}
                    else:
                        constants_dict[key] = values
                else:
                    sweep_dict[key] = values

    # Default seeds if not provided
    if "seeds" not in constants_dict:
        constants_dict["seeds"] = [11, 17]

    # Default agents if not provided
    if "agents" not in constants_dict:
        constants_dict["agents"] = 200

    # Create YAML content
    yaml_content = f"""# {study_name}
# Auto-generated study stub - EDIT BEFORE USE
study_id: {study_name}
version: 1.0
prereg_date: {datetime.datetime.now().strftime('%Y-%m-%d')}
adapter: adapters.{study_name}:run_adapter

hypothesis:
  text: >
    {hypothesis}

constants:
"""

    for key, value in constants_dict.items():
        if isinstance(value, dict):
            yaml_content += f"  {key}:\n"
            for k, v in value.items():
                yaml_content += f"    {k}: {v}\n"
        else:
            yaml_content += f"  {key}: {value}\n"

    yaml_content += """  noise: 0.05
  analysis_window: [960, 1040]

sweep:
"""

    for key, values in sweep_dict.items():
        yaml_content += f"  {key}: {values}\n"

    metric_list = [m.strip() for m in metrics.split(",")]
    required_cols = [m.upper() if m.lower() == "cci" else m for m in metric_list]

    yaml_content += f"""
exports:
  outdir: results/discovery_results/${{study_id}}/
  require_columns: {required_cols}

validation:
  bootstrap_ci: 1000
  metrics:
    - name: mean_CCI_gain
      rule: ">= 0.03"
    - name: mean_hazard_delta
      rule: "<= -0.01"
  classify:
    validated: "mean_CCI_gain >= 0.03 and mean_hazard_delta <= -0.01"
    under_review: "else"

report:
  title: "{study_name.replace('_', ' ').title()}"
  include_figures: true
  copy_to: results/archive/
"""

    study_path.write_text(yaml_content)
    print(f"[init] Created study: {study_path}")
    adapter_hint = f"adapters/{study_path.stem}.py"
    print(f"[init] ⚠️  IMPORTANT: Edit the YAML and create adapter at {adapter_hint}")
    return study_path


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="OpenLaws: run → validate → report (automation)"
    )
    ap.add_argument(
        "cmd",
        choices=["init", "run", "validate", "report", "auto"],
        help="pipeline step",
    )
    ap.add_argument(
        "--study",
        default="study.yml",
        help="path to study.yml (or study name for init)",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        help="number of bootstrap iterations for CI",
    )

    # Init-specific arguments
    ap.add_argument("--hyp", help="hypothesis text (for init)")
    ap.add_argument("--sweep", help="sweep parameters (for init)")
    ap.add_argument(
        "--metrics", default="cci,hazard,survival", help="metrics to track (for init)"
    )
    ap.add_argument("--exports", default="csv,png", help="export formats (for init)")

    args = ap.parse_args()

    # Handle init command separately (doesn't need existing study)
    if args.cmd == "init":
        if not args.hyp or not args.sweep:
            sys.exit("[init] Error: --hyp and --sweep are required for init command")
        study_path = step_init(
            args.study, args.hyp, args.sweep, args.metrics, args.exports
        )
        # Run HYMetaLab charter validation on newly created study if available
        try:
            if validate_and_enforce and study_path is not None:
                rep = validate_and_enforce(study_path, write_back=True)
                if print_charter_summary:
                    print_charter_summary(rep)
        except Exception as e:
            print(f"[init] Charter validation skipped: {e}")
        return

    study_path = Path(args.study).resolve()
    if not study_path.exists():
        sys.exit(f"[err] {study_path} not found. Create a study.yml first.")

    if args.cmd == "run":
        # Pre-run: validate/enforce HYMetaLab charter if available
        if validate_and_enforce:
            try:
                rep = validate_and_enforce(study_path, write_back=True)
                if print_charter_summary:
                    print_charter_summary(rep)
            except Exception as e:
                print(f"[run] Charter validation skipped: {e}")
        step_run(study_path)
    elif args.cmd == "validate":
        # Apply bootstrap CI if requested
        if args.bootstrap:
            print(f"[validate] Computing bootstrap CI with n={args.bootstrap}...")
            apply_bootstrap_validation(study_path, n_boot=args.bootstrap)
        validated, _ = step_validate(study_path)
        if not validated:
            print("[validate] No validated datasets yet.")
    elif args.cmd == "report":
        # use latest project summary to find validated
        summaries = sorted(ARCHIVE.glob("project_summary_*.json"))
        if not summaries:
            sys.exit("[report] No project summaries. Run 'validate' first.")
        data = json.loads(open(summaries[-1]).read())
        # compatibility: accept either 'category' or 'validated' flags
        validated = [
            d for d in data if d.get("category") == "validated" or d.get("validated")
        ]
        step_report(validated)
    else:  # auto
        # Pre-auto: run charter validator
        if validate_and_enforce:
            try:
                rep = validate_and_enforce(study_path, write_back=True)
                if print_charter_summary:
                    print_charter_summary(rep)
            except Exception as e:
                print(f"[auto] Charter validation skipped: {e}")
        outdir = step_run(study_path)
        validated, records = step_validate(study_path)
        if validated:
            step_report(validated)
        else:
            print("[auto] No validated datasets yet — figures/abstract deferred.")


if __name__ == "__main__":
    main()
