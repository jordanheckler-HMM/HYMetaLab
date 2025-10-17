import argparse
import time
from pathlib import Path

from .dtw_fast import build_epoch_matrix, dtw_epoch_matrix, save_heatmap
from .granger_fast import granger_two_var, save_granger_heatmap
from .plotting_fast import save_example_timeseries
from .recurrence_fast import analyze, analyze_epochs
from .report_fast import build_report
from .simulate import synthesize_run
from .surrogates_fast import surrogate_rqa_compare


def ensure_dirs(base):
    base = Path(base)
    (base / "metrics").mkdir(parents=True, exist_ok=True)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    (base / "report").mkdir(parents=True, exist_ok=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--shock_every", type=int, default=4800)
    parser.add_argument("--outputs", type=str, default="outputs_fast")
    parser.add_argument("--n_surrogates", type=int, default=30)
    parser.add_argument("--tau_percentile", type=int, default=10)
    parser.add_argument("--max_lag", type=int, default=4)
    parser.add_argument("--dtw_mode", type=str, default="pca1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    out = ensure_dirs(args.outputs)

    cfg = vars(args)
    # simulate
    t1 = time.time()
    df = synthesize_run(
        steps=args.steps,
        epochs=args.epochs,
        shock_every=args.shock_every,
        seed=args.seed,
    )
    if args.profile:
        print("simulate:", time.time() - t1)

    # global RQA
    t2 = time.time()
    rqa_global, R_global, tau = analyze(
        df, tau_percentile=args.tau_percentile, outdir=str(out / "figures")
    )
    if args.profile:
        print("rqa global:", time.time() - t2)
    # save rqa_global
    import csv

    with open(out / "metrics" / "rqa_global.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in rqa_global.items():
            writer.writerow([k, v])

    # per-epoch RQA
    t3 = time.time()
    rqa_epochs_df = analyze_epochs(
        df, tau_percentile=args.tau_percentile, outdir=str(out / "figures")
    )
    rqa_epochs_df.to_csv(out / "metrics" / "rqa_epochs.csv", index=False)
    if args.profile:
        print("rqa epochs:", time.time() - t3)

    # DTW
    t4 = time.time()
    groups = build_epoch_matrix(df)
    dtw_mat = dtw_epoch_matrix(groups, mode=args.dtw_mode)
    import numpy as np

    np.savetxt(out / "metrics" / "dtw_epoch_distance.csv", dtw_mat, delimiter=",")
    save_heatmap(dtw_mat, str(out / "figures" / "dtw_heatmap.png"))
    if args.profile:
        print("dtw:", time.time() - t4)

    # Surrogates (global only)
    t5 = time.time()
    sur_df = surrogate_rqa_compare(
        df,
        lambda d: analyze(
            d, tau_percentile=args.tau_percentile, outdir=str(out / "figures")
        ),
        n=args.n_surrogates,
        seed=args.seed,
    )
    sur_df.to_csv(out / "metrics" / "surrogate_comparison.csv", index=False)
    if args.profile:
        print("surrogates:", time.time() - t5)

    # Granger
    t6 = time.time()
    gr_res = granger_two_var(df, max_lag=args.max_lag)
    import csv

    with open(out / "metrics" / "granger_results.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "p", "stat", "selected_lag"])
        writer.writerow(
            [
                "CCI->Rc",
                gr_res.get("CCI_to_Rc_p"),
                gr_res.get("CCI_to_Rc_stat"),
                gr_res.get("selected_lag"),
            ]
        )
        writer.writerow(
            [
                "Rc->CCI",
                gr_res.get("Rc_to_CCI_p"),
                gr_res.get("Rc_to_CCI_stat"),
                gr_res.get("selected_lag"),
            ]
        )
    save_granger_heatmap(gr_res, str(out / "figures" / "granger_heatmap.png"))
    if args.profile:
        print("granger:", time.time() - t6)

    # plotting example timeseries
    save_example_timeseries(df, str(out / "figures" / "example_timeseries.png"))

    # report
    t7 = time.time()
    rep_path = build_report(
        cfg, rqa_global, rqa_epochs_df, dtw_mat, gr_res, outdir=str(out / "report")
    )
    if args.profile:
        print("report:", time.time() - t7)

    if args.profile:
        print("total:", time.time() - t0)
    print("Done. Outputs under", str(out))


if __name__ == "__main__":
    main()
