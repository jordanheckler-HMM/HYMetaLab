"""Convenience runner: execute smoke_run, validate the run, and produce a quick analysis plot."""

import os

from examples.quick_analysis import plot_avg_confidence
from sim.validate_run import validate_run
from smoke_run import tiny_simulation


def run_example():
    out = tiny_simulation()
    run_dir = os.path.dirname(out["manifest"])
    val = validate_run(run_dir)
    print("Validation summary:", val)
    decisions = os.path.join(run_dir, "decisions.jsonl")
    plot_avg_confidence(
        decisions, os.path.join(run_dir, "quick_plots", "avg_confidence.png")
    )


if __name__ == "__main__":
    run_example()
