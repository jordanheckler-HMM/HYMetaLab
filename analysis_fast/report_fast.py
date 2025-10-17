import json
from pathlib import Path


def _fmt(v, fmt_str=":.4g"):
    try:
        if v is None:
            return "NA"
        return ("{" + fmt_str + "}").format(v)
    except Exception:
        return str(v)


def build_report(
    cfg, rqa_global, rqa_epochs_df, dtw_mat, granger_res, outdir="outputs_fast/report"
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# Fast Analysis Report\n")
    md.append("## Configuration\n")
    md.append("```json\n" + json.dumps(cfg, indent=2) + "\n```\n")

    md.append("## RQA Global\n")
    md.append("|metric|value|")
    md.append("|---|---|")
    for k, v in (rqa_global or {}).items():
        md.append(f"|{k}|{v}|")

    md.append("\n## RQA per-epoch\n")
    # avoid dependency on pandas.to_markdown/tabulate
    try:
        txt = rqa_epochs_df.to_string(index=False)
    except Exception:
        txt = str(rqa_epochs_df)
    md.append("```\n" + txt + "\n```")

    md.append("\n## DTW matrix\n")
    md.append("Epoch×Epoch DTW matrix:")
    # format matrix rows
    try:
        rows = ["  ".join([f"{x:.3f}" for x in row]) for row in dtw_mat]
        md.append("```\n" + "\n".join(rows) + "\n```")
    except Exception:
        md.append("```\n" + str(dtw_mat) + "\n```")

    md.append("\n## Granger Results\n")
    md.append("|test|p|stat|selected_lag|")
    md.append("|---:|---:|---:|---:|")
    cci_p = granger_res.get("CCI_to_Rc_p") if granger_res else None
    cci_stat = granger_res.get("CCI_to_Rc_stat") if granger_res else None
    rc_p = granger_res.get("Rc_to_CCI_p") if granger_res else None
    rc_stat = granger_res.get("Rc_to_CCI_stat") if granger_res else None
    sel_lag = granger_res.get("selected_lag") if granger_res else None
    md.append(f"|CCI->Rc|{_fmt(cci_p)}|{_fmt(cci_stat)}|{_fmt(sel_lag,'')}|")
    md.append(f"|Rc->CCI|{_fmt(rc_p)}|{_fmt(rc_stat)}|{_fmt(sel_lag,'')}|")

    md.append("\n## Figures\n")
    md.append("![](../figures/recurrence_plot_GLOBAL.png)")
    try:
        n = len(rqa_epochs_df)
    except Exception:
        n = 0
    for i in range(n):
        md.append(f"![](../figures/recurrence_plot_EPOCH_{i}.png)")
    md.append("![](../figures/dtw_heatmap.png)")
    md.append("![](../figures/granger_heatmap.png)")
    md.append("![](../figures/example_timeseries.png)")

    md.append("\n## Conclusions\n")
    md.append(
        "- Quick summary: global RR={:.3f}, DET={:.3f}".format(
            rqa_global.get("RR", 0), rqa_global.get("DET", 0)
        )
    )
    md.append("- DTW indicates distances between epochs (see heatmap).")
    md.append(
        f"- Granger causality p-values suggest: CCI->Rc p={_fmt(cci_p)}, Rc->CCI p={_fmt(rc_p)}."
    )

    md.append("\n## Next steps\n")
    md.append("- Increase surrogates to test stability of RQA z-scores.")
    md.append("- Run Transfer Entropy or multivariate causality if needed.")

    p = outdir / "report.md"
    p.write_text("\n\n".join(md))
    return str(p)
