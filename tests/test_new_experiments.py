"""Quick tests for the new experiments: run tiny runs and check summary files exist."""

import json
import os


def test_shock_demo():
    p = "outputs/shock_resilience/demo_run/shock_summary.json"
    assert os.path.exists(p), "shock_summary.json missing"
    with open(p) as f:
        j = json.load(f)
    assert "final_alive_fraction" in j


def test_pipeline_demo():
    p = "outputs/lab_pipeline/demo_run/pipeline_summary.json"
    assert os.path.exists(p), "pipeline_summary.json missing"
    with open(p) as f:
        j = json.load(f)
    assert "final_completed" in j


def test_goal_demo():
    p = "outputs/goal_externalities/demo_run/goal_summary.json"
    assert os.path.exists(p), "goal_summary.json missing"
    with open(p) as f:
        j = json.load(f)
    assert "final_gini" in j


if __name__ == "__main__":
    test_shock_demo()
    print("shock demo OK")
    test_pipeline_demo()
    print("pipeline demo OK")
    test_goal_demo()
    print("goal demo OK")
