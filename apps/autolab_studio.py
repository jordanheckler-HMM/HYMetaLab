from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

# Ensure repo root is importable even when launched from a desktop bundle.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.autolab_ui_core import (
    agent_insight_summary,
    build_hypothesis,
    call_ollama,
    choose_output_folder,
    command_for_experiment,
    create_starter_project,
    ensure_workspace,
    finalize_background_run,
    get_run_detail,
    list_ollama_models,
    list_runs,
    load_agents,
    load_experiments,
    load_settings,
    local_review_answer,
    make_agent,
    make_experiment,
    open_in_file_browser,
    poll_background_run,
    persist_run,
    run_agent_stage,
    save_agents,
    save_experiments,
    save_settings,
    setup_progress,
    start_background_run,
    stop_background_run,
)

DEFAULT_WORKSPACE = str(
    Path(os.environ.get("AUTOLAB_WORKSPACE", str(Path.cwd() / "autolab_workspace")))
    .expanduser()
    .resolve()
)


def _agent_label(agent: dict[str, Any]) -> str:
    return f"{agent.get('name', 'agent')} ({agent.get('role', 'role')}) [{agent.get('model', 'model')}]"


def _trim(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]"


def _ensure_session_defaults() -> None:
    st.session_state.setdefault("workspace_path", DEFAULT_WORKSPACE)
    st.session_state.setdefault("ollama_models", [])
    st.session_state.setdefault("ollama_error", None)
    st.session_state.setdefault("active_run", None)
    st.session_state.setdefault("active_run_meta", None)
    st.session_state.setdefault("last_run_record", None)


def _read_text_file(path_text: str, max_chars: int = 6000) -> str:
    try:
        text = Path(path_text).read_text()
    except OSError:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def main() -> None:
    st.set_page_config(page_title="HYMetaLab AutoLab Studio", layout="wide")
    _ensure_session_defaults()

    st.title("HYMetaLab AutoLab Studio")
    st.caption("Single-user AutoLab MVP: setup -> agents -> experiments -> run -> review")

    with st.sidebar:
        st.subheader("Workspace")
        workspace_input = st.text_input(
            "Output Folder",
            value=st.session_state["workspace_path"],
            help="All UI-managed experiment artifacts are stored here.",
        )
        col_ws_a, col_ws_b = st.columns(2)
        with col_ws_a:
            if st.button("Use This Folder", use_container_width=True):
                st.session_state["workspace_path"] = workspace_input
                st.rerun()
        with col_ws_b:
            if st.button("Browse...", use_container_width=True):
                selected, error = choose_output_folder(st.session_state["workspace_path"])
                if error:
                    st.error(f"Folder picker unavailable: {error}")
                elif selected:
                    st.session_state["workspace_path"] = selected
                    st.rerun()

        workspace = ensure_workspace(st.session_state["workspace_path"])
        st.success(f"Active: {workspace}")
        col_open_a, col_open_b = st.columns(2)
        with col_open_a:
            if st.button("Open Folder", use_container_width=True):
                error = open_in_file_browser(workspace)
                if error:
                    st.error(f"Could not open folder: {error}")
        with col_open_b:
            if st.button("Open Runs", use_container_width=True):
                error = open_in_file_browser(workspace / "runs")
                if error:
                    st.error(f"Could not open runs: {error}")
        st.divider()
        st.caption("Tip: keep one workspace per project for clean history.")

    settings = load_settings(workspace)
    settings["workspace_path"] = str(workspace)
    save_settings(workspace, settings)
    agents = load_agents(workspace)
    experiments = load_experiments(workspace)
    if not st.session_state.get("ollama_models") and settings.get("cached_models"):
        st.session_state["ollama_models"] = list(settings.get("cached_models", []))

    progress = setup_progress(settings=settings, agents=agents, experiments=experiments)
    with st.sidebar:
        st.subheader("Setup Progress")
        st.progress(progress["ready_count"] / progress["total_steps"])
        st.caption(f"{progress['ready_count']}/{progress['total_steps']} steps complete")

    tabs = st.tabs(
        [
            "0) Wizard",
            "1) Setup",
            "2) Agents",
            "3) Experiments",
            "4) Run",
            "5) History",
            "6) Review Chat",
        ]
    )

    with tabs[0]:
        st.subheader("First-Run Wizard")
        if progress["all_ready"]:
            st.success("Setup complete. You can run experiments.")
        else:
            st.warning("Complete the remaining steps before running experiments.")

        for step in progress["steps"]:
            marker = "✅" if step["ready"] else "⬜"
            st.write(f"{marker} {step['label']}")

        if not progress["steps"][1]["ready"]:
            st.info("Next action: open Setup tab and run 'Check Ollama Connection'.")
        elif not progress["steps"][2]["ready"]:
            st.info("Next action: create at least one Agent.")
        elif not progress["steps"][3]["ready"]:
            st.info("Next action: create at least one Experiment template.")
        else:
            st.info("Next action: go to Run tab and execute your first experiment.")

        st.markdown(
            """
            **How this works**
            - Agents run first (Ollama prompts per selected agent).
            - Their outputs are saved to each run.
            - The experiment command is then executed and logged.
            - You review all results in History and Review Chat.
            """
        )

        st.divider()
        st.markdown("### Quick Start Presets")
        default_model = settings.get("default_chat_model")
        if not default_model and st.session_state.get("ollama_models"):
            default_model = st.session_state["ollama_models"][0]
        if st.button("Create Starter Project", use_container_width=True):
            starter = create_starter_project(default_model=default_model or "llama3.1:8b")
            existing_agent_names = {a.get("name") for a in agents}
            existing_exp_names = {e.get("name") for e in experiments}

            new_agents = [a for a in starter["agents"] if a.get("name") not in existing_agent_names]
            new_experiments = [e for e in starter["experiments"] if e.get("name") not in existing_exp_names]
            agents.extend(new_agents)
            experiments.extend(new_experiments)
            save_agents(workspace, agents)
            save_experiments(workspace, experiments)
            st.success(
                f"Starter project ready. Added {len(new_agents)} agent(s) and {len(new_experiments)} experiment template(s)."
            )
            st.rerun()

    with tabs[1]:
        st.subheader("Setup")
        ollama_url = st.text_input(
            "Ollama URL",
            value=settings.get("ollama_url", "http://127.0.0.1:11434"),
            help="Default local endpoint for Ollama.",
        )
        col_a, col_b = st.columns([1, 1])
        with col_a:
            if st.button("Check Ollama Connection", use_container_width=True):
                models, error = list_ollama_models(ollama_url)
                st.session_state["ollama_models"] = models
                st.session_state["ollama_error"] = error
                settings["ollama_url"] = ollama_url
                settings["ollama_connected"] = error is None
                settings["cached_models"] = models
                save_settings(workspace, settings)
        with col_b:
            if st.button("Refresh Installed Models", use_container_width=True):
                models, error = list_ollama_models(ollama_url)
                st.session_state["ollama_models"] = models
                st.session_state["ollama_error"] = error
                settings["cached_models"] = models
                save_settings(workspace, settings)

        if st.session_state["ollama_error"]:
            st.error(f"Ollama connection failed: {st.session_state['ollama_error']}")

        models = st.session_state.get("ollama_models", [])
        if models:
            st.success(f"Connected. {len(models)} installed model(s) found.")
            default_chat_model = settings.get("default_chat_model", models[0])
            selected_model = st.selectbox(
                "Default Review Chat Model",
                options=models,
                index=models.index(default_chat_model)
                if default_chat_model in models
                else 0,
            )
            if st.button("Save Chat Model", use_container_width=True):
                settings["default_chat_model"] = selected_model
                settings["ollama_url"] = ollama_url
                settings["cached_models"] = models
                save_settings(workspace, settings)
                st.success("Default chat model saved.")
        else:
            st.info("No models loaded yet. Start Ollama and run connection check.")

        st.markdown(
            """
            **First-run checklist**
            1. Set output folder.
            2. Check Ollama connection.
            3. Create at least one agent.
            4. Create at least one experiment template.
            5. Run and review.
            """
        )

    with tabs[2]:
        st.subheader("Agents")
        st.caption("Each agent maps to an Ollama model + role prompt.")

        with st.form("create_agent_form", clear_on_submit=True):
            name = st.text_input("Agent Name", placeholder="Hypothesis Builder")
            role = st.text_input("Role", placeholder="Generates clear hypotheses")
            system_prompt = st.text_area(
                "System Prompt",
                placeholder="You generate concise, testable experiment hypotheses.",
                height=120,
            )
            model_options = st.session_state.get("ollama_models", [])
            if model_options:
                model = st.selectbox("Model", options=model_options)
            else:
                model = st.text_input("Model", placeholder="llama3.1:8b")

            submitted = st.form_submit_button("Create Agent", use_container_width=True)
            if submitted:
                if not name.strip() or not role.strip() or not model.strip():
                    st.error("Name, role, and model are required.")
                else:
                    agents.append(make_agent(name=name, role=role, model=model, system_prompt=system_prompt))
                    save_agents(workspace, agents)
                    st.success("Agent created.")

        st.divider()
        if not agents:
            st.info("No agents yet.")
        else:
            st.markdown("### Current Agents")
            for idx, agent in enumerate(agents):
                with st.expander(_agent_label(agent), expanded=False):
                    st.write(f"**Prompt:** {agent.get('system_prompt', '') or '(empty)'}")
                    if st.button("Delete Agent", key=f"delete_agent_{agent.get('id', idx)}"):
                        agents = [a for a in agents if a.get("id") != agent.get("id")]
                        save_agents(workspace, agents)
                        st.rerun()

    with tabs[3]:
        st.subheader("Experiments")
        st.caption("Simple templates for non-technical run setup.")

        with st.form("create_experiment_form", clear_on_submit=True):
            exp_name = st.text_input("Experiment Name", placeholder="Stability Sweep")
            objective = st.text_area(
                "Objective",
                placeholder="Find settings that maximize stability while staying within Guardian thresholds.",
                height=100,
            )
            hypothesis = st.text_area(
                "Hypothesis Seed",
                placeholder="Lower noise and higher coherence should increase pass rates.",
                height=100,
            )
            mode = st.selectbox(
                "Run Mode",
                options=["guided", "autopilot"],
                help="guided: one direct run via lab_techs_runner.py | autopilot: multi-cycle via auto_lab.py",
            )
            cycles = st.number_input(
                "Autopilot Cycles",
                min_value=1,
                value=3,
                step=1,
                help="Only used for autopilot mode.",
            )
            create_exp = st.form_submit_button("Save Experiment Template", use_container_width=True)
            if create_exp:
                if not exp_name.strip():
                    st.error("Experiment name is required.")
                else:
                    experiments.append(
                        make_experiment(
                            name=exp_name,
                            objective=objective,
                            hypothesis=hypothesis,
                            mode=mode,
                            autopilot_cycles=int(cycles),
                        )
                    )
                    save_experiments(workspace, experiments)
                    st.success("Experiment template saved.")

        st.divider()
        if not experiments:
            st.info("No experiments yet.")
        else:
            st.markdown("### Saved Templates")
            for idx, exp in enumerate(experiments):
                label = f"{exp.get('name', 'experiment')} ({exp.get('mode', 'guided')})"
                with st.expander(label, expanded=False):
                    st.write(f"**Objective:** {exp.get('objective', '') or '(empty)'}")
                    st.write(f"**Hypothesis:** {exp.get('hypothesis', '') or '(empty)'}")
                    st.write(f"**Autopilot Cycles:** {exp.get('autopilot_cycles', 1)}")
                    if st.button("Delete Template", key=f"delete_exp_{exp.get('id', idx)}"):
                        experiments = [e for e in experiments if e.get("id") != exp.get("id")]
                        save_experiments(workspace, experiments)
                        st.rerun()

    with tabs[4]:
        st.subheader("Run")
        active_run = st.session_state.get("active_run")
        active_run_meta = st.session_state.get("active_run_meta")
        if active_run:
            status = poll_background_run(active_run)
            st.markdown("### Active Run")
            st.write(f"PID: {status.get('pid')} | Elapsed: {status.get('elapsed_sec')} sec")
            st.code(" ".join(active_run.get("command", [])), language="bash")

            live_stdout = _read_text_file(str(active_run.get("stdout_path", "")), max_chars=4000)
            live_stderr = _read_text_file(str(active_run.get("stderr_path", "")), max_chars=2000)
            if live_stdout:
                st.text_area("Live Stdout (tail)", value=live_stdout, height=160)
            if live_stderr.strip():
                st.text_area("Live Stderr (tail)", value=live_stderr, height=120)

            col_status_a, col_status_b = st.columns(2)
            with col_status_a:
                if st.button("Refresh Status", use_container_width=True):
                    st.rerun()
            with col_status_b:
                if st.button("Stop Run", use_container_width=True):
                    stop_background_run(active_run)
                    st.warning("Run stop requested. Refresh status to finalize.")

            if not status.get("running"):
                run_result = finalize_background_run(active_run)
                meta = active_run_meta or {}
                record = persist_run(
                    workspace=workspace,
                    experiment=meta.get("experiment", {}),
                    selected_agents=meta.get("selected_agents", []),
                    run_result=run_result,
                    agent_outputs=meta.get("agent_outputs", []),
                )
                st.session_state["last_run_record"] = record
                st.session_state["active_run"] = None
                st.session_state["active_run_meta"] = None
                st.success(f"Run complete and saved as {record['run_id']}.")
                st.rerun()

        if not progress["all_ready"]:
            st.warning("Wizard incomplete. Finish Setup, Agents, and Experiments first.")
        elif not experiments:
            st.warning("Create an experiment template first.")
        elif active_run:
            st.info("A run is in progress. Wait for completion before starting another run.")
        else:
            exp_options = {exp["name"]: exp for exp in experiments}
            selected_exp_name = st.selectbox("Experiment Template", options=list(exp_options.keys()))
            selected_exp = exp_options[selected_exp_name]

            selected_agent_labels = st.multiselect(
                "Agents",
                options=[_agent_label(a) for a in agents],
                help="Choose which agents are attached to this run context.",
            )
            selected_agents = [a for a in agents if _agent_label(a) in selected_agent_labels]
            hypothesis_text = build_hypothesis(selected_exp, selected_agents)
            st.text_area("Generated Hypothesis Context", value=hypothesis_text, height=140, disabled=True)
            run_agent_planning = st.checkbox(
                "Run agent planning stage before experiment",
                value=True,
                help="Calls each selected agent's Ollama model and saves outputs in run artifacts.",
            )

            command = command_for_experiment(selected_exp, hypothesis_text)
            st.code(" ".join(command), language="bash")

            if st.button("Start Run", type="primary", use_container_width=True):
                agent_outputs: list[dict[str, Any]] = []
                with st.spinner("Preparing run..."):
                    if run_agent_planning and selected_agents:
                        agent_outputs = run_agent_stage(
                            api_base=settings.get("ollama_url", "http://127.0.0.1:11434"),
                            experiment=selected_exp,
                            agents=selected_agents,
                        )
                        insight = agent_insight_summary(agent_outputs)
                        if insight and selected_exp.get("mode") != "autopilot":
                            enriched = f"{hypothesis_text} | agent_insights: {insight}"
                            command = command_for_experiment(selected_exp, enriched)
                    live_state = start_background_run(
                        workspace=workspace,
                        command=command,
                    )
                    st.session_state["active_run"] = live_state
                    st.session_state["active_run_meta"] = {
                        "experiment": selected_exp,
                        "selected_agents": selected_agents,
                        "agent_outputs": agent_outputs,
                    }
                st.success("Run started. Use Refresh Status to poll progress.")
                st.rerun()

    with tabs[5]:
        st.subheader("History")
        runs = list_runs(workspace)
        if not runs:
            st.info("No runs yet.")
        else:
            st.dataframe(runs, use_container_width=True)
            run_id = st.selectbox("Open Run Detail", options=[r["run_id"] for r in runs])
            if st.button("Open Selected Run Folder", use_container_width=True):
                error = open_in_file_browser(workspace / "runs" / run_id)
                if error:
                    st.error(f"Could not open run folder: {error}")
            detail = get_run_detail(workspace, run_id)
            if detail:
                st.markdown(f"### Run {run_id}")
                st.json(
                    {
                        "status": detail.get("status"),
                        "experiment": detail.get("experiment"),
                        "agents": detail.get("agents"),
                        "agent_output_count": detail.get("agent_output_count", 0),
                        "command": detail.get("command"),
                        "returncode": detail.get("returncode"),
                        "duration_sec": detail.get("duration_sec"),
                        "started_at": detail.get("started_at"),
                    }
                )
                if detail.get("agent_outputs"):
                    st.markdown("**Agent Planning Outputs**")
                    st.json(detail.get("agent_outputs"))
                st.text_area("Stdout", value=_trim(detail.get("stdout", "")), height=220)
                if detail.get("stderr", "").strip():
                    st.text_area("Stderr", value=_trim(detail.get("stderr", "")), height=140)

    with tabs[6]:
        st.subheader("Review Chat")
        runs = list_runs(workspace)
        if not runs:
            st.info("Run at least one experiment first.")
        else:
            run_ids = [r["run_id"] for r in runs]
            selected_run_ids = st.multiselect("Runs to Review", options=run_ids, default=run_ids[:3])
            question = st.text_area(
                "Question",
                placeholder="What failed most often, and what should I run next?",
                height=100,
            )

            use_ollama = st.checkbox("Use Ollama for answer generation", value=True)
            ollama_url = settings.get("ollama_url", "http://127.0.0.1:11434")
            model_options = st.session_state.get("ollama_models", [])
            model = settings.get("default_chat_model", "")
            if model_options:
                model = st.selectbox(
                    "Review Model",
                    options=model_options,
                    index=model_options.index(model) if model in model_options else 0,
                )

            if st.button("Generate Review", use_container_width=True):
                chosen_runs = [r for r in runs if r["run_id"] in selected_run_ids]
                fallback = local_review_answer(question=question, selected_runs=chosen_runs)
                if not use_ollama:
                    st.markdown("### Review")
                    st.write(fallback)
                else:
                    details = [get_run_detail(workspace, run_id) for run_id in selected_run_ids]
                    details = [d for d in details if d]
                    context_blocks = []
                    for detail in details:
                        context_blocks.append(
                            json_block(
                                detail.get("run_id", ""),
                                detail.get("status", ""),
                                detail.get("stdout", ""),
                                detail.get("agent_outputs", []),
                            )
                        )
                    prompt = (
                        "You are an experiment review assistant. Use only the provided run context.\n"
                        "Return: (1) key outcomes, (2) failures, (3) next run suggestion.\n\n"
                        f"Question: {question or 'Summarize these runs.'}\n\n"
                        "Run context:\n"
                        + "\n\n".join(context_blocks)
                    )
                    answer, error = call_ollama(api_base=ollama_url, model=model, prompt=prompt)
                    st.markdown("### Review")
                    if error:
                        st.warning(f"Ollama call failed: {error}")
                        st.write("Fallback summary:")
                        st.write(fallback)
                    elif not answer.strip():
                        st.write("Ollama returned an empty answer. Fallback summary:")
                        st.write(fallback)
                    else:
                        st.write(answer)


def json_block(run_id: str, status: str, stdout: str, agent_outputs: list[dict[str, Any]]) -> str:
    insight = agent_insight_summary(agent_outputs, max_chars=300)
    return (
        f"run_id={run_id}\n"
        f"status={status}\n"
        f"agent_insights={insight or 'none'}\n"
        f"stdout:\n{_trim(stdout, max_chars=2500)}"
    )


if __name__ == "__main__":
    main()
