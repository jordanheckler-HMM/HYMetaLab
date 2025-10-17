#!/usr/bin/env python3
"""Compute per-feeling metric proxies from run outputs and summaries.

Produces discovery_results/meaning_table_inputs/feeling_effects_v03.csv

Strategy:
- For runs with discovery summary CSVs (meaning_phase5_4), extract M_rec, T_rec, disp_rec_mean as proxies.
- For runs with raw outputs (outputs/run_*), compute from time_series.csv, lifespans.csv, culture.jsonl, integration.jsonl.
- Build consistent metrics: CCI_mean, hazard_mean, collapse_risk_mean, survival_mean, and per-feeling means Tr_mean, Hp_mean, T_mean, M_mean.
"""
import glob
import json
import math
from pathlib import Path

import pandas as pd


def read_summary_variant(variant):
    # search discovery_results for summary_{variant}.csv
    patterns = [
        f"discovery_results/**/summary_{variant}*.csv",
        f"discovery_results/**/{variant}*/summary*.csv",
    ]
    files = []
    for p in patterns:
        files.extend(glob.glob(p, recursive=True))
    files = sorted(set(files))
    if not files:
        return None
    return pd.read_csv(files[0])


def compute_from_outputs(outdir: Path):
    # compute proxies from raw outputs
    ts = outdir / "time_series.csv"
    lf = outdir / "lifespans.csv"
    cult = outdir / "culture.jsonl"
    integ = outdir / "integration.jsonl"

    res = {}
    if ts.exists():
        df = pd.read_csv(ts)
        res["CCI_mean"] = (
            float(df["avg_consciousness"].mean())
            if "avg_consciousness" in df.columns
            else float("nan")
        )
        res["Hp_mean"] = (
            float(df["avg_innovation"].mean())
            if "avg_innovation" in df.columns
            else float("nan")
        )
        res["min_num_agents"] = (
            int(df["num_agents"].min()) if "num_agents" in df.columns else None
        )
        res["mean_num_agents"] = (
            float(df["num_agents"].mean()) if "num_agents" in df.columns else None
        )
    else:
        res["CCI_mean"] = math.nan
        res["Hp_mean"] = math.nan

    if lf.exists():
        ldf = pd.read_csv(lf)
        # survival_mean: mean survival time normalized by ticks if ticks column isn't present use max
        if "survival_time" in ldf.columns:
            res["survival_mean"] = float(ldf["survival_time"].mean())
        else:
            res["survival_mean"] = math.nan
    else:
        res["survival_mean"] = math.nan

    # culture.jsonl: compute mean reputation and cumulative trust change
    if cult.exists():
        reput = {}
        trust_changes = []
        for line in cult.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            aid = obj.get("agent_id")
            reput[aid] = obj.get("reputation", reput.get(aid, 0.0))
            trust_changes.append(float(obj.get("trust_change", 0.0)))
        res["Tr_mean"] = float(sum(reput.values()) / len(reput)) if reput else math.nan
        res["Tr_cum_change_mean"] = (
            float(sum(trust_changes) / len(trust_changes))
            if trust_changes
            else math.nan
        )
    else:
        res["Tr_mean"] = math.nan
        res["Tr_cum_change_mean"] = math.nan

    # integration.jsonl: compute average conflicts per tick as hazard proxy
    if integ.exists():
        conflicts = []
        ticks = set()
        for line in integ.read_text().splitlines():
            try:
                obj = json.loads(line)
            except Exception:
                continue
            conflicts.append(int(obj.get("conflicts", 0)))
            ticks.add(int(obj.get("tick", -1)))
        res["hazard_mean"] = (
            float(sum(conflicts) / len(conflicts)) if conflicts else math.nan
        )
    else:
        res["hazard_mean"] = math.nan

    # collapse risk: 1 - min_num_agents / initial (approx)
    if res.get("min_num_agents") is not None and res.get("mean_num_agents") is not None:
        # try to infer initial agents from mean or max
        init = max(int(res["mean_num_agents"]), int(res["min_num_agents"]))
        if init > 0:
            res["collapse_risk_mean"] = 1.0 - (res["min_num_agents"] / init)
        else:
            res["collapse_risk_mean"] = math.nan
    else:
        res["collapse_risk_mean"] = math.nan

    return res


def build_variant_record(variant_name, summary_df=None, outputs_dir=None):
    rec = {"variant": variant_name}
    # initialize fields
    fields = [
        "CCI_mean",
        "hazard_mean",
        "collapse_risk_mean",
        "survival_mean",
        "Tr_mean",
        "Hp_mean",
        "T_mean",
        "M_mean",
    ]
    for f in fields:
        rec[f] = float("nan")

    out_res = {}
    if outputs_dir and outputs_dir.exists():
        out_res = compute_from_outputs(outputs_dir)

    # Fill from summary if present (prefer summary for M/T related values)
    if summary_df is not None and not summary_df.empty:
        s = summary_df.iloc[0]
        # M_mean use M_rec if available else DeltaM
        if "M_rec" in s.index:
            rec["M_mean"] = float(s.get("M_rec", float("nan")))
        elif "DeltaM" in s.index:
            rec["M_mean"] = float(s.get("DeltaM", float("nan")))
        # T_mean use T_rec
        if "T_rec" in s.index:
            rec["T_mean"] = float(s.get("T_rec", float("nan")))
        # dispersion as hazard proxy
        if "disp_rec_mean" in s.index:
            rec["hazard_mean"] = float(s.get("disp_rec_mean", float("nan")))

    # overlay outputs-derived values
    for k, v in out_res.items():
        if k in rec and (math.isnan(rec[k]) or rec[k] is None):
            rec[k] = v
        else:
            # map some keys
            if k == "Tr_mean":
                rec["Tr_mean"] = v
            if k == "Hp_mean":
                rec["Hp_mean"] = v
            if k == "CCI_mean" and math.isnan(rec["CCI_mean"]):
                rec["CCI_mean"] = v
            if k == "hazard_mean" and (
                math.isnan(rec["hazard_mean"]) or rec["hazard_mean"] is None
            ):
                rec["hazard_mean"] = v
            if k == "collapse_risk_mean" and math.isnan(rec["collapse_risk_mean"]):
                rec["collapse_risk_mean"] = v
            if k == "survival_mean" and math.isnan(rec["survival_mean"]):
                rec["survival_mean"] = v

    # derive T_mean if still missing: average of CCI and predictability if available
    if math.isnan(rec["T_mean"]):
        vals = []
        if not math.isnan(rec.get("CCI_mean", float("nan"))):
            vals.append(rec["CCI_mean"])
        # try to read predictability_summary.csv from outputs_dir
        if outputs_dir is not None:
            ps = outputs_dir / "predictability_summary.csv"
            if ps.exists():
                try:
                    pdf = pd.read_csv(ps)
                    if "accuracy" in pdf.columns:
                        vals.append(float(pdf["accuracy"].mean()))
                except Exception:
                    pass
        if vals:
            rec["T_mean"] = float(sum(vals) / len(vals))

    # derive M_mean composite if missing
    if math.isnan(rec["M_mean"]):
        parts = []
        for src in ["T_mean", "Hp_mean", "Tr_mean"]:
            if not math.isnan(rec.get(src, float("nan"))):
                parts.append(rec[src])
        if parts:
            rec["M_mean"] = float(
                (
                    0.4 * (rec.get("T_mean", 0))
                    + 0.3 * (rec.get("Hp_mean", 0))
                    + 0.3 * (rec.get("Tr_mean", 0))
                )
                if parts
                else float("nan")
            )

    return rec


def main():
    variants = ["baseline_v54", "anneal_v1", "anneal_v2", "anneal_adaptive"]
    records = []

    # find most recent outputs run for adaptive
    # assume latest run_1234_2000t_80a is adaptive run
    adaptive_output = None
    outs = sorted(
        glob.glob("outputs/run_*"), key=lambda p: Path(p).stat().st_mtime, reverse=True
    )
    if outs:
        adaptive_output = Path(outs[0])

    for v in variants:
        summary = read_summary_variant(v)
        outputs_dir = None
        if v == "anneal_adaptive" and adaptive_output:
            outputs_dir = adaptive_output
        rec = build_variant_record(v, summary_df=summary, outputs_dir=outputs_dir)
        records.append(rec)

    # build feeling rows
    feelings = {
        "Trust": [
            "Tr_mean",
            "CCI_mean",
            "hazard_mean",
            "collapse_risk_mean",
            "survival_mean",
        ],
        "Hope": [
            "Hp_mean",
            "CCI_mean",
            "hazard_mean",
            "collapse_risk_mean",
            "survival_mean",
        ],
        "Truth": [
            "T_mean",
            "CCI_mean",
            "hazard_mean",
            "collapse_risk_mean",
            "survival_mean",
        ],
        "Meaning": [
            "M_mean",
            "CCI_mean",
            "hazard_mean",
            "collapse_risk_mean",
            "survival_mean",
        ],
    }

    df = pd.DataFrame(records)
    # set variant as study column
    df["study"] = df["variant"]

    # compute baseline/adaptive means
    baseline = df[df["study"].str.contains("baseline", case=False)].mean(
        numeric_only=True
    )
    adaptive = df[df["study"].str.contains("adaptive", case=False)].mean(
        numeric_only=True
    )

    out_rows = []
    for feeling, cols in feelings.items():
        for metric in cols:
            if metric not in df.columns:
                continue
            base_val = baseline.get(metric, float("nan"))
            adapt_val = adaptive.get(metric, float("nan"))
            delta = adapt_val - base_val
            direction = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            out_rows.append(
                {
                    "feeling": feeling,
                    "metric": metric,
                    "baseline_mean": base_val,
                    "adaptive_mean": adapt_val,
                    "delta": delta,
                    "effect_direction": direction,
                }
            )

    outdir = Path("discovery_results/meaning_table_inputs")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "feeling_effects_v03.csv"
    pd.DataFrame(out_rows).to_csv(outpath, index=False)
    print("Wrote", outpath)


if __name__ == "__main__":
    main()
