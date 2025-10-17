import os

from experiments.lesion_experiment import run_lesion
from experiments.survival_experiment import run_survival


def test_lesion_quick(tmp_path):
    out = run_lesion(
        n_agents=10,
        ticks=5,
        seeds=[123],
        workspace_enabled_values=[True, False],
        export_base=str(tmp_path),
    )
    assert os.path.exists(out["run_dir"])
    assert len(out["results"]) == 2


def test_lesion_stats_quick(tmp_path):
    from experiments.lesion_with_stats import run_lesion_with_stats

    out = run_lesion_with_stats(
        n_agents=10,
        ticks=5,
        seeds=[123],
        workspace_enabled_values=[True, False],
        export_base=str(tmp_path),
    )
    assert os.path.exists(out["stats"])


def test_survival_quick(tmp_path):
    out = run_survival(
        n_subjects=20,
        max_time=10,
        treatment_effect=0.2,
        seed=123,
        export_base=str(tmp_path),
    )
    assert os.path.exists(out["run_dir"])
    assert os.path.exists(out["summary"])
