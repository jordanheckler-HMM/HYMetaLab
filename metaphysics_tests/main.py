"""CLI entrypoint for metaphysics_tests
Usage: python -m metaphysics_tests.main
"""

import argparse
import logging
from pathlib import Path

import yaml

from . import (
    causality,
    dtw_utils,
    io_utils,
    plotting,
    preprocess,
    recurrence,
    report,
    surrogates,
)

logger = logging.getLogger("metaphysics_tests")


def load_config(path="config.yaml"):
    p = Path(path)
    if p.exists():
        return yaml.safe_load(p.read_text())
    # defaults
    return {
        "paths": {"input_glob": "data/*.csv", "outputs_dir": "outputs"},
        "recurrence": {"tau_strategy": "percentile", "tau_percentile": 10},
        "surrogates": {"n": 100, "kinds": ["phase", "shuffle"]},
        "granger": {"max_lag": 12, "ic": "aic", "fdr_alpha": 0.05},
        "plotting": {"dpi": 220},
        "report": {"title": "Patterns of Consciousness", "author": "Heck Yeah Lab"},
    }


def setup_logging(outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "logs").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(outdir / "logs" / "run.log")
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


def synth_demo_and_run(outputs_dir, cfg):
    # produce small demo and run analyses
    df = preprocess.synthesize_demo()
    return run_pipeline(df, outputs_dir, cfg)


def run_pipeline(df, outputs_dir, cfg):
    out = Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    # ensure subfolders
    for sub in ["figures", "metrics", "json", "report", "logs"]:
        (out / sub).mkdir(parents=True, exist_ok=True)
    # ensure columns
    df = io_utils.normalize_cols(df)
    df = io_utils.infer_epochs(df)
    cols = ["cci", "meaning", "coherence", "rc", "epsilon"]
    # zscore
    if cfg.get("preprocess", {}).get("zscore", True):
        df = preprocess.zscore_df(df, [c for c in cols if c in df.columns])
    epochs = preprocess.slice_epochs(df)

    summaries = {}
    figures = []

    # Recurrence global
    X = preprocess.build_state_matrix if False else None
    # assemble multivariate X over whole run
    avail = [c for c in cols if c in df.columns]
    Xall = df[avail].astype(float).values
    R, tau = recurrence.recurrence_matrix(
        Xall,
        tau="percentile",
        tau_percentile=cfg.get("recurrence", {}).get("tau_percentile", 10),
    )
    recurrence.recurrence_plot(
        R,
        outpath=out / "figures" / "recurrence_plot_GLOBAL.png",
        dpi=cfg.get("plotting", {}).get("dpi", 220),
        title="Global RP",
    )
    summaries["recurrence_global_tau"] = tau
    figures.append("figures/recurrence_plot_GLOBAL.png")
    # global RQA
    rqa = recurrence.rqa_metrics(R, min_diag=cfg.get("rqa", {}).get("min_diag_line", 2))
    summaries["rqa_global"] = rqa
    # per-epoch RQA (limit to first 12)
    per_epoch = {}
    for i, (e, edf) in enumerate(epochs.items()):
        if i >= 12:
            break
        X = edf[avail].astype(float).values
        R_e, _ = recurrence.recurrence_matrix(
            X,
            tau="percentile",
            tau_percentile=cfg.get("recurrence", {}).get("tau_percentile", 10),
        )
        rqa_e = recurrence.rqa_metrics(
            R_e, min_diag=cfg.get("rqa", {}).get("min_diag_line", 2)
        )
        per_epoch[e] = rqa_e
        recurrence.recurrence_plot(
            R_e,
            outpath=out / "figures" / f"recurrence_plot_EPOCH_{e}.png",
            dpi=cfg.get("plotting", {}).get("dpi", 220),
            title=f"RP Epoch {e}",
        )
        figures.append(f"figures/recurrence_plot_EPOCH_{e}.png")
    summaries["rqa_epochs"] = per_epoch

    # DTW: reduce per-epoch to PCA1 and compute distances
    series_list = []
    for e, edf in epochs.items():
        comp = (
            dtw_utils.pca1_reduce(edf, avail)
            if len(edf) > 1
            else edf[avail].astype(float).mean(axis=0)
        )
        series_list.append(comp)
    M = dtw_utils.dtw_matrix(series_list, window=cfg.get("dtw", {}).get("window", None))
    summaries["dtw_matrix_shape"] = M.shape
    plotting.save_heatmap(
        M, out / "figures" / "dtw_heatmap.png", title="DTW epoch distances"
    )
    figures.append("figures/dtw_heatmap.png")

    # Surrogates: compute RQA metric RR for surrogates
    n = cfg.get("surrogates", {}).get("n", 100)
    # generate small N (for speed) if demo
    n = min(n, 100)
    sur_stats = []
    for kind in cfg.get("surrogates", {}).get("kinds", ["phase", "shuffle"]):
        # do 20 surrogates for speed
        for i in range(min(20, n)):
            # per-channel phase surrogate
            Y = df[avail].astype(float).values.copy()
            if kind == "phase":
                for c in range(Y.shape[1]):
                    Y[:, c] = surrogates.phase_randomized(Y[:, c])
            else:
                for c in range(Y.shape[1]):
                    Y[:, c] = surrogates.shuffle_surrogate(Y[:, c])
            R_s, _ = recurrence.recurrence_matrix(
                Y,
                tau="percentile",
                tau_percentile=cfg.get("recurrence", {}).get("tau_percentile", 10),
            )
            rqa_s = recurrence.rqa_metrics(
                R_s, min_diag=cfg.get("rqa", {}).get("min_diag_line", 2)
            )
            sur_stats.append((kind, rqa_s))
    summaries["surrogates_sample"] = str(sur_stats[:5])

    # Granger
    try:
        gr = causality.granger_pairwise(
            df[avail].dropna(),
            avail,
            maxlag=cfg.get("granger", {}).get("max_lag", 12),
            ic=cfg.get("granger", {}).get("ic", "aic"),
        )
        if gr is not None:
            gr.to_csv(out / "metrics" / "granger_results.csv", index=False)
            figures.append("figures/granger_heatmap.png")
    except Exception as e:
        logger.exception("Granger step failed: %s", e)

    # report
    mdpath, pdfpath = report.build_report(out, cfg, summaries, figures)

    print("\nDone. Outputs under", out)
    print("Key files:")
    print(" - Recurrence global plot:", out / "figures" / "recurrence_plot_GLOBAL.png")
    print(" - DTW heatmap:", out / "figures" / "dtw_heatmap.png")
    print(" - Report:", mdpath, pdfpath)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", default=None)
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    setup_logging(args.outputs_dir)
    if args.input_glob:
        df = io_utils.load_csvs(args.input_glob)
    else:
        df = None
    if df is None:
        print("No input: synthesizing demo data")
        df = preprocess.synthesize_demo()
    run_pipeline(df, args.outputs_dir, cfg)


if __name__ == "__main__":
    main()
