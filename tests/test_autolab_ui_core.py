from __future__ import annotations

import sys
import time
from pathlib import Path

from apps.autolab_ui_core import (
    agent_insight_summary,
    build_hypothesis,
    choose_output_folder,
    create_starter_project,
    ensure_workspace,
    finalize_background_run,
    list_runs,
    make_agent,
    make_experiment,
    open_in_file_browser,
    poll_background_run,
    persist_run,
    run_agent_stage,
    start_background_run,
    setup_progress,
)


def test_ensure_workspace_bootstraps_expected_files(tmp_path: Path):
    workspace = ensure_workspace(tmp_path / "studio_workspace")

    assert (workspace / "agents.json").exists()
    assert (workspace / "experiments.json").exists()
    assert (workspace / "settings.json").exists()
    assert (workspace / "runs" / "index.json").exists()


def test_build_hypothesis_includes_experiment_and_agent_context():
    experiment = make_experiment(
        name="Stability Sweep",
        objective="Find stable settings",
        hypothesis="Higher coherence should improve pass rate.",
    )
    agent = make_agent(
        name="Planner",
        role="Experiment planning",
        model="llama3.1:8b",
        system_prompt="Create clear hypotheses.",
    )

    prompt = build_hypothesis(experiment, [agent])

    assert "Stability Sweep" in prompt
    assert "Find stable settings" in prompt
    assert "Higher coherence should improve pass rate." in prompt
    assert "Planner" in prompt
    assert "llama3.1:8b" in prompt


def test_persist_run_writes_artifacts_and_index(tmp_path: Path):
    workspace = ensure_workspace(tmp_path / "studio_workspace")
    experiment = make_experiment(
        name="Quick Run",
        objective="Smoke test",
        hypothesis="Baseline should pass.",
        mode="guided",
    )
    agent = make_agent(
        name="Operator",
        role="Runs tests",
        model="llama3.1:8b",
        system_prompt="Operate safely.",
    )
    run_result = {
        "command": ["python", "lab_techs_runner.py", "--hypothesis", "baseline"],
        "returncode": 0,
        "stdout": "✅ Guardian PASS. Result:",
        "stderr": "",
        "timed_out": False,
        "started_at": "2026-01-01T00:00:00Z",
        "finished_at": "2026-01-01T00:00:02Z",
        "duration_sec": 2.0,
    }

    record = persist_run(
        workspace=workspace,
        experiment=experiment,
        selected_agents=[agent],
        run_result=run_result,
        agent_outputs=[{"agent_name": "Operator", "response": "Try baseline first.", "error": None}],
    )

    run_dir = workspace / "runs" / record["run_id"]
    assert run_dir.exists()
    assert (run_dir / "run.json").exists()
    assert (run_dir / "stdout.txt").exists()
    assert (run_dir / "stderr.txt").exists()
    assert (run_dir / "agent_outputs.json").exists()
    assert record["agent_output_count"] == 1

    rows = list_runs(workspace)
    assert len(rows) == 1
    assert rows[0]["status"] == "pass"
    assert rows[0]["experiment_name"] == "Quick Run"


def test_setup_progress_tracks_readiness():
    settings = {"workspace_path": "/tmp/example", "ollama_connected": True}
    agents = [{"id": "a1"}]
    experiments = [{"id": "e1"}]

    progress = setup_progress(settings=settings, agents=agents, experiments=experiments)

    assert progress["all_ready"] is True
    assert progress["ready_count"] == progress["total_steps"]


def test_run_agent_stage_uses_each_agent_model(monkeypatch):
    captured_calls = []

    def fake_call_ollama(api_base: str, model: str, prompt: str):
        captured_calls.append((api_base, model, prompt))
        return f"response-from-{model}", None

    monkeypatch.setattr("apps.autolab_ui_core.call_ollama", fake_call_ollama)

    experiment = make_experiment(
        name="Agent Stage",
        objective="Find robust setup",
        hypothesis="Higher coherence should improve pass rate.",
    )
    agents = [
        make_agent("Planner", "planning", "model-a", "Produce one candidate."),
        make_agent("Reviewer", "review", "model-b", "Flag one risk."),
    ]

    outputs = run_agent_stage(api_base="http://127.0.0.1:11434", experiment=experiment, agents=agents)

    assert len(outputs) == 2
    assert len(captured_calls) == 2
    assert outputs[0]["response"] == "response-from-model-a"
    assert outputs[1]["response"] == "response-from-model-b"

    summary = agent_insight_summary(outputs, max_chars=30)
    assert "Planner:" in summary
    assert "Reviewer:" in summary


def test_create_starter_project_returns_agents_and_experiments():
    starter = create_starter_project(default_model="llama3.1:8b")

    assert len(starter["agents"]) >= 2
    assert len(starter["experiments"]) >= 2
    assert starter["agents"][0]["model"] == "llama3.1:8b"


def test_background_run_lifecycle(tmp_path: Path):
    workspace = ensure_workspace(tmp_path / "studio_workspace")
    command = [sys.executable, "-c", "print('hello-from-live-run')"]

    live_state = start_background_run(workspace=workspace, command=command)
    assert live_state["pid"] is not None

    # Poll briefly until finished
    for _ in range(20):
        status = poll_background_run(live_state)
        if not status["running"]:
            break
        time.sleep(0.05)

    result = finalize_background_run(live_state)
    assert result["returncode"] == 0
    assert "hello-from-live-run" in result["stdout"]


def test_choose_output_folder_handles_missing_gui(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tkinter" or name.startswith("tkinter"):
            raise RuntimeError("tkinter unavailable in test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    selected, error = choose_output_folder()
    assert selected is None
    assert error is not None
    assert "unavailable" in error


def test_open_in_file_browser_errors_for_missing_path(tmp_path: Path):
    missing = tmp_path / "missing-folder"
    error = open_in_file_browser(missing)
    assert error is not None
    assert "does not exist" in error


def test_open_in_file_browser_runs_platform_command(monkeypatch, tmp_path: Path):
    captured: list[list[str]] = []

    class _DummyProc:
        pass

    def fake_popen(command):
        captured.append(command)
        return _DummyProc()

    monkeypatch.setattr("apps.autolab_ui_core.subprocess.Popen", fake_popen)

    error = open_in_file_browser(tmp_path)
    assert error is None
    assert captured
    assert captured[0][0] in {"open", "explorer", "xdg-open"}
