import os

from experiments.calibration_experiment import run_calibration
from experiments.plot_calibration import plot_calibration


def test_calibration_pipeline(tmp_path):
    # Use a temporary config by copying baseline and adjusting params inline
    cfg_path = "configs/calibration.yaml"
    res = run_calibration(cfg_path)
    # Check results exist
    run_dir = res["run_dir"]
    assert os.path.exists(run_dir)
    # find one summary and plot
    any_key = next(iter(res["results"]))
    summary = res["results"][any_key]["summary"]
    assert os.path.exists(summary)
    out_png = os.path.join(os.path.dirname(summary), "calib.png")
    plot_calibration(summary, out_png)
    assert os.path.exists(out_png)
