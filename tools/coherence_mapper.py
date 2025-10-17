#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

SUMMARY_GLOB = "discovery_results/*/summary.json"
OUT_HTML = Path("tools/cross_domain_map.html")


def load_df() -> pd.DataFrame:
    rows = []
    for p in Path(".").glob(SUMMARY_GLOB):
        try:
            d = json.loads(p.read_text())
            d["_source_dir"] = str(p.parent)
            rows.append(d)
        except:
            pass
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows, max_level=2)
    # Normalize column names
    ren = {}
    for c in df.columns:
        lc = c.lower()
        if "delta_cci" in lc or "Δcci" in c:
            ren[c] = "delta_cci"
        elif "delta_hazard" in lc or ("hazard" in lc and "delta" in lc):
            ren[c] = "delta_hazard"
        elif lc.endswith(".epsilon") or lc == "epsilon":
            ren[c] = "epsilon"
        elif ".rho" in lc or lc == "rho":
            ren[c] = "rho"
        elif "openlaws_score" in lc:
            ren[c] = "openlaws_score"
    df = df.rename(columns=ren)
    keep = [
        "_source_dir",
        "delta_cci",
        "delta_hazard",
        "epsilon",
        "rho",
        "openlaws_score",
    ]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    for k in keep:
        if k != "_source_dir":
            df[k] = pd.to_numeric(df[k], errors="coerce")
    df["lab"] = (
        df["_source_dir"]
        .str.extract(r"discovery_results/([^/]+)/")[0]
        .fillna("unknown")
    )
    return df


def export_html(df: pd.DataFrame):
    if df.empty:
        OUT_HTML.write_text("<h3>No summaries found.</h3>")
        return
    # 2D scatter (rho vs epsilon colored by ΔCCI)
    fig1 = px.scatter(
        df,
        x="rho",
        y="epsilon",
        color="delta_cci",
        hover_data=["_source_dir", "delta_hazard", "openlaws_score"],
        title="Coherence Surface (ΔCCI color)",
    )
    fig2 = px.scatter(
        df,
        x="rho",
        y="epsilon",
        color="delta_hazard",
        hover_data=["_source_dir", "delta_cci", "openlaws_score"],
        title="Hazard Surface (Δhazard color)",
    )
    # Simple HTML bundle
    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Cross-Domain Coherence Map</title></head>
    <body style="font-family: sans-serif; margin:24px;">
      <h1>Cross-Domain Coherence Map</h1>
      <p>Auto-generated from discovery_results/*/summary.json</p>
      {fig1.to_html(include_plotlyjs="cdn", full_html=False)}
      <hr/>
      {fig2.to_html(include_plotlyjs=False, full_html=False)}
    </body>
    </html>
    """
    OUT_HTML.write_text(html)


def main():
    df = load_df()
    export_html(df)
    print(f"Wrote {OUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
