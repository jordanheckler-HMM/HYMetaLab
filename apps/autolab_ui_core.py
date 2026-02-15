from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]


def _iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def ensure_workspace(path: str | Path) -> Path:
    workspace = Path(path).expanduser().resolve()
    (workspace / "runs").mkdir(parents=True, exist_ok=True)

    defaults: list[tuple[Path, Any]] = [
        (workspace / "agents.json", []),
        (workspace / "experiments.json", []),
        (workspace / "settings.json", {}),
        (workspace / "runs" / "index.json", []),
    ]

    for file_path, default_payload in defaults:
        if not file_path.exists():
            _write_json(file_path, default_payload)

    return workspace


def choose_output_folder(initial_dir: str | Path | None = None) -> tuple[str | None, str | None]:
    """
    Open a native directory picker (when available) and return the selected path.
    """
    start_dir = str(Path(initial_dir).expanduser()) if initial_dir else str(Path.home())
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(
            initialdir=start_dir,
            title="Select AutoLab Output Folder",
            mustexist=False,
        )
        root.destroy()
        if not selected:
            return None, None
        return str(Path(selected).expanduser().resolve()), None
    except Exception as exc:
        return None, str(exc)


def open_in_file_browser(path: str | Path) -> str | None:
    target = Path(path).expanduser().resolve()
    if not target.exists():
        return f"Path does not exist: {target}"

    if sys.platform == "darwin":
        command = ["open", str(target)]
    elif os.name == "nt":
        command = ["explorer", str(target)]
    else:
        command = ["xdg-open", str(target)]

    try:
        subprocess.Popen(command)
    except OSError as exc:
        return str(exc)
    return None


def load_agents(workspace: Path) -> list[dict[str, Any]]:
    return _read_json(workspace / "agents.json", [])


def save_agents(workspace: Path, agents: list[dict[str, Any]]) -> None:
    _write_json(workspace / "agents.json", agents)


def load_experiments(workspace: Path) -> list[dict[str, Any]]:
    return _read_json(workspace / "experiments.json", [])


def save_experiments(workspace: Path, experiments: list[dict[str, Any]]) -> None:
    _write_json(workspace / "experiments.json", experiments)


def load_settings(workspace: Path) -> dict[str, Any]:
    return _read_json(workspace / "settings.json", {})


def save_settings(workspace: Path, settings: dict[str, Any]) -> None:
    _write_json(workspace / "settings.json", settings)


def setup_progress(
    settings: dict[str, Any],
    agents: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
) -> dict[str, Any]:
    steps = [
        {
            "id": "output_folder",
            "label": "Output folder selected",
            "ready": bool(settings.get("workspace_path")),
        },
        {
            "id": "ollama",
            "label": "Ollama connected",
            "ready": bool(settings.get("ollama_connected")),
        },
        {
            "id": "agents",
            "label": "At least one agent created",
            "ready": len(agents) > 0,
        },
        {
            "id": "experiments",
            "label": "At least one experiment template created",
            "ready": len(experiments) > 0,
        },
    ]
    ready_count = sum(1 for step in steps if step["ready"])
    return {
        "steps": steps,
        "ready_count": ready_count,
        "total_steps": len(steps),
        "all_ready": ready_count == len(steps),
    }


def list_runs(workspace: Path) -> list[dict[str, Any]]:
    runs = _read_json(workspace / "runs" / "index.json", [])
    return sorted(runs, key=lambda row: row.get("started_at", ""), reverse=True)


def get_run_detail(workspace: Path, run_id: str) -> dict[str, Any] | None:
    record_path = workspace / "runs" / run_id / "run.json"
    if not record_path.exists():
        return None
    detail = _read_json(record_path, {})
    run_dir = record_path.parent
    detail["stdout"] = (run_dir / "stdout.txt").read_text() if (run_dir / "stdout.txt").exists() else ""
    detail["stderr"] = (run_dir / "stderr.txt").read_text() if (run_dir / "stderr.txt").exists() else ""
    detail["agent_outputs"] = _read_json(run_dir / "agent_outputs.json", [])
    return detail


def list_ollama_models(api_base: str) -> tuple[list[str], str | None]:
    base = api_base.rstrip("/")
    try:
        response = requests.get(f"{base}/api/tags", timeout=5)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        names = [m.get("name", "").strip() for m in models if m.get("name")]
        return sorted(set(names)), None
    except requests.RequestException as exc:
        return [], str(exc)


def call_ollama(api_base: str, model: str, prompt: str) -> tuple[str, str | None]:
    base = api_base.rstrip("/")
    body = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(f"{base}/api/generate", json=body, timeout=60)
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip(), None
    except requests.RequestException as exc:
        return "", str(exc)


def make_agent(name: str, role: str, model: str, system_prompt: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "role": role.strip(),
        "model": model.strip(),
        "system_prompt": system_prompt.strip(),
        "created_at": _iso_now(),
    }


def make_experiment(
    name: str,
    objective: str,
    hypothesis: str,
    mode: str = "guided",
    autopilot_cycles: int = 3,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "name": name.strip(),
        "objective": objective.strip(),
        "hypothesis": hypothesis.strip(),
        "mode": mode.strip(),
        "autopilot_cycles": max(1, int(autopilot_cycles)),
        "created_at": _iso_now(),
    }


def create_starter_project(default_model: str) -> dict[str, list[dict[str, Any]]]:
    model = (default_model or "llama3.1:8b").strip()
    agents = [
        make_agent(
            name="Hypothesis Planner",
            role="Proposes concise test variants",
            model=model,
            system_prompt=(
                "Generate one focused hypothesis variant and one measurable success condition."
            ),
        ),
        make_agent(
            name="Risk Reviewer",
            role="Flags execution and quality risks",
            model=model,
            system_prompt=(
                "Identify the top execution risk and propose one mitigation before run."
            ),
        ),
    ]
    experiments = [
        make_experiment(
            name="Baseline Guided Run",
            objective="Establish a stable baseline with Guardian pass criteria.",
            hypothesis="A clear, narrow baseline should pass more consistently.",
            mode="guided",
            autopilot_cycles=1,
        ),
        make_experiment(
            name="Autopilot Exploration",
            objective="Explore nearby variants and compare pass/block outcomes.",
            hypothesis="Moderate exploration should surface at least one better-performing variant.",
            mode="autopilot",
            autopilot_cycles=3,
        ),
    ]
    return {"agents": agents, "experiments": experiments}


def build_hypothesis(experiment: dict[str, Any], agents: list[dict[str, Any]]) -> str:
    lines = [
        f"experiment: {experiment.get('name', 'untitled')}",
        f"objective: {experiment.get('objective', '').strip() or 'none provided'}",
        f"hypothesis: {experiment.get('hypothesis', '').strip() or 'none provided'}",
    ]
    if agents:
        agent_lines = []
        for agent in agents:
            agent_lines.append(
                f"{agent.get('name', 'agent')} ({agent.get('role', 'role unknown')}, model={agent.get('model', 'unknown')})"
            )
        lines.append("agents: " + "; ".join(agent_lines))
    return " | ".join(lines)


def build_agent_prompt(agent: dict[str, Any], experiment: dict[str, Any]) -> str:
    system_prompt = agent.get("system_prompt", "").strip()
    role = agent.get("role", "assistant")
    name = agent.get("name", "agent")
    objective = experiment.get("objective", "").strip() or "No objective provided."
    hypothesis = experiment.get("hypothesis", "").strip() or "No hypothesis provided."
    return (
        f"Agent: {name}\n"
        f"Role: {role}\n"
        f"Instruction: {system_prompt or 'Provide practical experiment guidance.'}\n\n"
        f"Experiment name: {experiment.get('name', 'untitled')}\n"
        f"Objective: {objective}\n"
        f"Hypothesis seed: {hypothesis}\n\n"
        "Return a concise plan with:\n"
        "1) one suggested test variant\n"
        "2) one risk to monitor\n"
        "3) one success criterion"
    )


def run_agent_stage(
    api_base: str,
    experiment: dict[str, Any],
    agents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for agent in agents:
        prompt = build_agent_prompt(agent, experiment)
        response, error = call_ollama(
            api_base=api_base,
            model=str(agent.get("model", "")),
            prompt=prompt,
        )
        outputs.append(
            {
                "agent_id": agent.get("id"),
                "agent_name": agent.get("name"),
                "role": agent.get("role"),
                "model": agent.get("model"),
                "prompt": prompt,
                "response": response,
                "error": error,
                "created_at": _iso_now(),
            }
        )
    return outputs


def agent_insight_summary(agent_outputs: list[dict[str, Any]], max_chars: int = 180) -> str:
    if not agent_outputs:
        return ""
    parts: list[str] = []
    for item in agent_outputs:
        name = str(item.get("agent_name") or "agent")
        response = str(item.get("response") or "")
        error = str(item.get("error") or "")
        if error:
            snippet = f"error={error}"
        else:
            snippet = response.replace("\n", " ").strip()
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars] + "..."
        parts.append(f"{name}: {snippet}")
    return " || ".join(parts)


def execute_run(command: list[str], timeout_sec: int = 1800) -> dict[str, Any]:
    started = datetime.now(UTC)
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        finished = datetime.now(UTC)
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "timed_out": False,
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "finished_at": finished.isoformat().replace("+00:00", "Z"),
            "duration_sec": round((finished - started).total_seconds(), 3),
        }
    except subprocess.TimeoutExpired as exc:
        finished = datetime.now(UTC)
        return {
            "command": command,
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
            "timed_out": True,
            "started_at": started.isoformat().replace("+00:00", "Z"),
            "finished_at": finished.isoformat().replace("+00:00", "Z"),
            "duration_sec": round((finished - started).total_seconds(), 3),
        }


def start_background_run(workspace: Path, command: list[str]) -> dict[str, Any]:
    live_id = str(uuid.uuid4())
    active_dir = workspace / "runs" / "_active" / live_id
    active_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = active_dir / "stdout.txt"
    stderr_path = active_dir / "stderr.txt"
    stdout_handle = stdout_path.open("w")
    stderr_handle = stderr_path.open("w")
    started = datetime.now(UTC)

    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=stdout_handle,
        stderr=stderr_handle,
        text=True,
    )
    return {
        "live_id": live_id,
        "command": command,
        "process": process,
        "pid": process.pid,
        "started_at": started.isoformat().replace("+00:00", "Z"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "stdout_handle": stdout_handle,
        "stderr_handle": stderr_handle,
    }


def poll_background_run(live_state: dict[str, Any]) -> dict[str, Any]:
    process = live_state.get("process")
    if process is None:
        return {"running": False, "returncode": 1, "elapsed_sec": 0.0}

    returncode = process.poll()
    started_at = datetime.fromisoformat(
        str(live_state.get("started_at", _iso_now())).replace("Z", "+00:00")
    )
    elapsed = round((datetime.now(UTC) - started_at).total_seconds(), 3)
    return {
        "running": returncode is None,
        "returncode": returncode,
        "elapsed_sec": elapsed,
        "pid": live_state.get("pid"),
    }


def stop_background_run(live_state: dict[str, Any]) -> None:
    process = live_state.get("process")
    if process is None:
        return
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _safe_close(handle: Any) -> None:
    try:
        handle.close()
    except Exception:
        return


def finalize_background_run(live_state: dict[str, Any]) -> dict[str, Any]:
    process = live_state.get("process")
    if process is None:
        return {
            "command": live_state.get("command", []),
            "returncode": 1,
            "stdout": "",
            "stderr": "Background process missing",
            "timed_out": False,
            "started_at": live_state.get("started_at", _iso_now()),
            "finished_at": _iso_now(),
            "duration_sec": 0.0,
        }

    if process.poll() is None:
        stop_background_run(live_state)

    _safe_close(live_state.get("stdout_handle"))
    _safe_close(live_state.get("stderr_handle"))

    stdout_path = Path(str(live_state.get("stdout_path", "")))
    stderr_path = Path(str(live_state.get("stderr_path", "")))
    stdout = stdout_path.read_text() if stdout_path.exists() else ""
    stderr = stderr_path.read_text() if stderr_path.exists() else ""

    started_at = datetime.fromisoformat(
        str(live_state.get("started_at", _iso_now())).replace("Z", "+00:00")
    )
    finished_at = datetime.now(UTC)
    return {
        "command": live_state.get("command", []),
        "returncode": process.returncode if process.returncode is not None else 1,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": False,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.isoformat().replace("+00:00", "Z"),
        "duration_sec": round((finished_at - started_at).total_seconds(), 3),
    }


def classify_run(result: dict[str, Any]) -> str:
    stdout = str(result.get("stdout", ""))
    if result.get("timed_out"):
        return "timeout"
    if "✅ Guardian PASS" in stdout:
        return "pass"
    if "Blocked by Guardian" in stdout or "❌ Blocked by Guardian" in stdout:
        return "blocked"
    if int(result.get("returncode", 1)) != 0:
        return "error"
    return "completed"


def persist_run(
    workspace: Path,
    experiment: dict[str, Any],
    selected_agents: list[dict[str, Any]],
    run_result: dict[str, Any],
    agent_outputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    run_dir = workspace / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    stdout_path = run_dir / "stdout.txt"
    stderr_path = run_dir / "stderr.txt"
    stdout_path.write_text(str(run_result.get("stdout", "")))
    stderr_path.write_text(str(run_result.get("stderr", "")))
    agent_outputs = agent_outputs or []
    agent_outputs_path = run_dir / "agent_outputs.json"
    _write_json(agent_outputs_path, agent_outputs)

    status = classify_run(run_result)
    record = {
        "run_id": run_id,
        "status": status,
        "experiment": {
            "id": experiment.get("id"),
            "name": experiment.get("name"),
            "mode": experiment.get("mode"),
            "autopilot_cycles": experiment.get("autopilot_cycles", 1),
        },
        "agents": [{"id": a.get("id"), "name": a.get("name"), "model": a.get("model")} for a in selected_agents],
        "command": run_result.get("command", []),
        "returncode": run_result.get("returncode"),
        "timed_out": run_result.get("timed_out"),
        "started_at": run_result.get("started_at"),
        "finished_at": run_result.get("finished_at"),
        "duration_sec": run_result.get("duration_sec"),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "agent_outputs_path": str(agent_outputs_path),
        "agent_output_count": len(agent_outputs),
        "created_at": _iso_now(),
    }
    _write_json(run_dir / "run.json", record)

    index_path = workspace / "runs" / "index.json"
    index = _read_json(index_path, [])
    index.append(
        {
            "run_id": run_id,
            "status": status,
            "experiment_name": experiment.get("name"),
            "mode": experiment.get("mode"),
            "started_at": run_result.get("started_at"),
            "duration_sec": run_result.get("duration_sec"),
        }
    )
    _write_json(index_path, index)
    return record


def local_review_answer(question: str, selected_runs: list[dict[str, Any]]) -> str:
    if not selected_runs:
        return "Select at least one run to review."

    pass_count = sum(1 for run in selected_runs if run.get("status") == "pass")
    blocked_count = sum(1 for run in selected_runs if run.get("status") == "blocked")
    timeout_count = sum(1 for run in selected_runs if run.get("status") == "timeout")
    error_count = sum(1 for run in selected_runs if run.get("status") == "error")

    lines = [
        f"Runs reviewed: {len(selected_runs)}",
        f"Pass: {pass_count}",
        f"Blocked: {blocked_count}",
        f"Timeout: {timeout_count}",
        f"Error: {error_count}",
    ]

    lower_question = question.lower().strip()
    if "best" in lower_question or "recommend" in lower_question:
        if pass_count > 0:
            lines.append("Recommendation: use the latest passing run as your baseline and iterate gradually.")
        else:
            lines.append("Recommendation: tighten hypothesis scope and rerun once to establish a stable baseline.")
    elif "failed" in lower_question or "why" in lower_question:
        lines.append(
            "Failure hint: blocked runs are typically Guardian threshold failures; error runs usually indicate command/runtime issues."
        )

    return "\n".join(lines)


def command_for_experiment(experiment: dict[str, Any], hypothesis: str) -> list[str]:
    mode = experiment.get("mode", "guided")
    if mode == "autopilot":
        cycles = max(1, int(experiment.get("autopilot_cycles", 3)))
        return [sys.executable, "auto_lab.py", str(cycles)]
    return [sys.executable, "lab_techs_runner.py", "--hypothesis", hypothesis]
